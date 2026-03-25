/*
 * SmolVLM multimodal runner for ExecuTorch.
 */

#include <fstream>
#include <chrono>
#include <memory>
#include <unordered_set>
#include <vector>

#include <gflags/gflags.h>

#include <executorch/extension/llm/runner/constants.h>
#include <executorch/extension/llm/runner/io_manager/io_manager.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/llm/runner/multimodal_prefiller.h>
#include <executorch/extension/llm/runner/multimodal_runner.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/text_token_generator.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/log.h>

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

DEFINE_string(model_path, "smolvlm_vulkan.pte", "Path to .pte model.");
DEFINE_string(
    vision_model_path,
    "",
    "Optional path to a vision_encoder-only .pte model for split-module execution.");
DEFINE_string(
    token_model_path,
    "",
    "Optional path to a token_embedding-only .pte model for split-module execution.");
DEFINE_string(
    text_model_path,
    "",
    "Optional path to a text_decoder-only .pte model for split-module execution.");
DEFINE_string(tokenizer_path, "tokenizer.json", "Path to tokenizer JSON.");
DEFINE_string(
    image_bin_path,
    "image_512_chw_f32.bin",
    "Path to raw float32 CHW image tensor.");
DEFINE_string(
    image_embeddings_bin_path,
    "",
    "Optional path to precomputed float32 image embeddings [1, seq_len, 960].");
DEFINE_string(
    prompt_embeddings_bin_path,
    "",
    "Optional path to precomputed float32 prompt embeddings [1, seq_len, 960].");
DEFINE_string(prompt, "Can you describe this image?", "Text prompt.");
DEFINE_double(temperature, 0.0f, "Sampling temperature.");
DEFINE_int32(max_new_tokens, 96, "Maximum new tokens.");
DEFINE_int32(cpu_threads, -1, "CPU thread count.");
DEFINE_int32(
    prewarm_context_len,
    96,
    "Sequence length used for Vulkan prewarm. Must cover the target run.");
DEFINE_bool(warmup, false, "Run one warmup generation.");
DEFINE_bool(skip_prewarm, false, "Skip Vulkan method prewarm.");
DEFINE_bool(
    skip_image_prewarm,
    false,
    "Skip prewarming vision_encoder and image-embedding decoder shapes.");
DEFINE_bool(
    run_vision_only,
    false,
    "Run only the vision_encoder method once and exit.");
DEFINE_bool(
    run_text_decoder_only,
    false,
    "Run only the text_decoder method once and exit.");
DEFINE_bool(
    decoder_use_token_embedding,
    true,
    "When running text_decoder only, source embeddings from token_embedding instead of zeros.");
DEFINE_int32(
    decoder_seq_len,
    5,
    "Sequence length for the text_decoder-only diagnostic.");
DEFINE_int32(
    decoder_start_pos,
    0,
    "Start position for the text_decoder-only diagnostic.");

namespace {

using ::executorch::extension::llm::Image;
using ::executorch::extension::llm::MultimodalInput;
using ::executorch::extension::llm::MultimodalPrefiller;
using ::executorch::extension::llm::MultimodalRunner;
using ::executorch::extension::llm::MultimodalDecoderRunner;
using ::executorch::extension::llm::Stats;
using ::executorch::extension::llm::TextDecoderRunner;
using ::executorch::extension::llm::TextTokenGenerator;
using ::executorch::extension::llm::IOManager;
using ::executorch::extension::Module;
using ::executorch::extension::llm::make_image_input;
using ::executorch::extension::llm::make_text_input;
using ::executorch::extension::TensorPtr;
using ::executorch::extension::llm::populate_start_pos_or_cache_position;
using ::executorch::extension::llm::convert_to_bfloat16;

constexpr int kImageSize = 512;
constexpr int kChannels = 3;

TensorPtr make_zero_tensor(
    std::vector<executorch::aten::SizesType> sizes,
    executorch::aten::ScalarType type);

class SplitTextDecoderRunner final : public TextDecoderRunner {
 public:
  SplitTextDecoderRunner(Module* token_module, Module* text_module)
      : TextDecoderRunner(
            text_module,
            nullptr,
            ::executorch::extension::llm::kTextModelMethod),
        token_module_(token_module),
        text_module_(text_module) {}

  ::executorch::runtime::Error load() override {
    ET_CHECK_OK_OR_RETURN_ERROR(
        token_module_->load_method(::executorch::extension::llm::kTokenEmbeddingMethod));
    ET_CHECK_OK_OR_RETURN_ERROR(
        text_module_->load_method(::executorch::extension::llm::kTextModelMethod));
    return ::executorch::runtime::Error::Ok;
  }

  bool is_method_loaded() override {
    return token_module_->is_method_loaded(::executorch::extension::llm::kTokenEmbeddingMethod) &&
        text_module_->is_method_loaded(::executorch::extension::llm::kTextModelMethod);
  }

  ::executorch::runtime::Result<executorch::aten::Tensor> step(
      TensorPtr& tokens,
      int64_t start_pos) override {
    auto token_embedding_result = token_module_->execute(
        ::executorch::extension::llm::kTokenEmbeddingMethod, tokens);
    ET_CHECK_OK_OR_RETURN_ERROR(token_embedding_result.error());
    return decode_embeddings(token_embedding_result.get()[0], start_pos);
  }

  ::executorch::runtime::Result<executorch::aten::Tensor> decode_embeddings(
      const ::executorch::runtime::EValue& embeddings,
      int64_t start_pos) {
    std::vector<int64_t> cache_positions;
    auto cache_position_result =
        populate_start_pos_or_cache_position(
            text_module_,
            start_pos,
            cache_positions,
            static_cast<int>(embeddings.toTensor().size(1)),
            ::executorch::extension::llm::kTextModelMethod);
    ET_CHECK_OK_OR_RETURN_ERROR(cache_position_result.error());
    auto outputs = text_module_->execute(
        ::executorch::extension::llm::kTextModelMethod,
        {embeddings, cache_position_result.get()});
    ET_CHECK_OK_OR_RETURN_ERROR(outputs.error());
    return outputs.get()[0].toTensor();
  }

 private:
  Module* token_module_;
  Module* text_module_;
};

MultimodalInput load_image_bin(const std::string& image_path) {
  const size_t numel = static_cast<size_t>(kChannels) * kImageSize * kImageSize;
  const std::streamsize expected_bytes =
      static_cast<std::streamsize>(numel * sizeof(float));

  std::ifstream file(image_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    ET_LOG(Error, "Failed to open image bin: %s", image_path.c_str());
    throw std::runtime_error("failed to open image bin");
  }

  const std::streamsize size = file.tellg();
  if (size != expected_bytes) {
    ET_LOG(
        Error,
        "Unexpected image bin size: got %ld bytes, expected %ld",
        size,
        expected_bytes);
    throw std::runtime_error("unexpected image bin size");
  }

  file.seekg(0, std::ios::beg);
  std::vector<float> buffer(numel);
  file.read(reinterpret_cast<char*>(buffer.data()), expected_bytes);
  file.close();

  Image image(std::move(buffer), kImageSize, kImageSize, kChannels);
  return make_image_input(std::move(image));
}

TensorPtr load_embeddings_bin(
    const std::string& embeddings_path,
    const char* description) {
  constexpr int64_t kHiddenSize = 960;
  constexpr size_t kHiddenBytes = static_cast<size_t>(kHiddenSize) * sizeof(float);

  std::ifstream file(embeddings_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    ET_LOG(
        Error,
        "Failed to open %s bin: %s",
        description,
        embeddings_path.c_str());
    throw std::runtime_error("failed to open embeddings bin");
  }

  const std::streamsize size = file.tellg();
  ET_CHECK_MSG(size > 0, "Embeddings file must be non-empty");
  ET_CHECK_MSG(
      size % static_cast<std::streamsize>(kHiddenBytes) == 0,
      "Embeddings file size must be divisible by hidden size");

  const auto seq_len = static_cast<int64_t>(size / static_cast<std::streamsize>(kHiddenBytes));
  file.seekg(0, std::ios::beg);

  auto embeddings = make_zero_tensor(
      {
          1,
          static_cast<executorch::aten::SizesType>(seq_len),
          static_cast<executorch::aten::SizesType>(kHiddenSize),
      },
      executorch::aten::ScalarType::Float);
  file.read(reinterpret_cast<char*>(embeddings->mutable_data_ptr<float>()), size);
  file.close();
  ET_LOG(
      Info,
      "Loaded precomputed %s with seq_len=%" PRId64,
      description,
      seq_len);
  return embeddings;
}

class CloningMultimodalPrefiller final : public MultimodalPrefiller {
 public:
  using MultimodalPrefiller::MultimodalPrefiller;

  executorch::runtime::Result<uint64_t> prefill(
      const MultimodalInput& input,
      int64_t& start_pos,
      int32_t bos = 0,
      int32_t eos = 0) override {
    if (input.is_text() || input.is_tokens()) {
      std::vector<uint64_t> tokens;
      if (input.is_text()) {
        auto& text = input.get_text();
        auto encode_result = tokenizer_->encode(text, bos, eos);
        if (!encode_result.ok()) {
          ET_LOG(
              Error,
              "Tokenizers error code %d",
              static_cast<uint32_t>(encode_result.error()));
          return ::executorch::runtime::Error::InvalidArgument;
        }
        tokens = std::move(*encode_result);
      } else {
        tokens = input.get_tokens();
      }

      auto text_tensor = ::executorch::extension::from_blob(
          tokens.data(),
          {1, static_cast<executorch::aten::SizesType>(tokens.size())},
          ::executorch::aten::ScalarType::Long);
      ET_LOG(
          Info,
          "Runtime prefill: text/tokens len=%zu start_pos=%" PRId64,
          tokens.size(),
          start_pos);

      auto token_embedding_result =
          module_->execute(::executorch::extension::llm::kTokenEmbeddingMethod, text_tensor);
      ET_CHECK_OK_OR_RETURN_ERROR(token_embedding_result.error());
      auto token_embedding_outputs = token_embedding_result.get();
      auto encoder_output = ::executorch::extension::clone_tensor_ptr(
          token_embedding_outputs[0].toTensor());

      int64_t seq_len = encoder_output->size(1);
      std::vector<int64_t> cache_positions;
      auto cache_position_result = populate_start_pos_or_cache_position(
          module_,
          start_pos,
          cache_positions,
          static_cast<int>(seq_len),
          ::executorch::extension::llm::kTextModelMethod);
      ET_CHECK_OK_OR_RETURN_ERROR(cache_position_result.error());
      auto cache_position_tensor = cache_position_result.get();
      ET_LOG(
          Info,
          "Runtime prefill: text_decoder text/tokens seq_len=%" PRId64
          " start_pos=%" PRId64,
          seq_len,
          start_pos);

      auto prefill_result = module_->execute(
          ::executorch::extension::llm::kTextModelMethod,
          {encoder_output, cache_position_tensor});
      ET_CHECK_OK_OR_RETURN_ERROR(prefill_result.error());
      auto prefill_outputs = prefill_result.get();
      ET_CHECK_OR_RETURN_ERROR(
          !prefill_outputs.empty(),
          InvalidState,
          "Text encoder returned empty results when used to prefill decoder");

      start_pos += seq_len;
      return static_cast<uint64_t>(
          text_decoder_runner_->logits_to_token(prefill_outputs[0].toTensor()));
    }

    if (!input.is_image()) {
      return MultimodalPrefiller::prefill(input, start_pos, bos, eos);
    }

    const Image& image = input.get_image();
    auto method_meta_result =
        module_->method_meta(::executorch::extension::llm::kVisionEncoderMethod);
    ET_CHECK_OK_OR_RETURN_ERROR(
        method_meta_result.error(),
        "Failed to get method_meta for %s",
        ::executorch::extension::llm::kVisionEncoderMethod);
    auto method_meta = method_meta_result.get();

    ET_CHECK_OR_RETURN_ERROR(
        method_meta.num_inputs() > 0,
        InvalidArgument,
        "Image encoder should have at least 1 input");
    auto input_meta_result = method_meta.input_tensor_meta(0);
    ET_CHECK_OK_OR_RETURN_ERROR(
        input_meta_result.error(), "Cannot get input tensor meta at index 0");
    auto input_meta = input_meta_result.get();
    auto expected_dtype = input_meta.scalar_type();

    if (expected_dtype == ::executorch::aten::ScalarType::Float) {
      ET_CHECK_OR_RETURN_ERROR(
          image.is_float(),
          InvalidArgument,
          "Model expects float image data, but image has uint8_t data.");
    } else if (expected_dtype == ::executorch::aten::ScalarType::Byte) {
      ET_CHECK_OR_RETURN_ERROR(
          image.is_uint8(),
          InvalidArgument,
          "Model expects uint8_t image data, but image has float data.");
    } else if (expected_dtype == ::executorch::aten::ScalarType::BFloat16) {
      ET_CHECK_OR_RETURN_ERROR(
          image.is_float(),
          InvalidArgument,
          "Model expects BFloat16 image data, but image has uint8_t data.");
    } else {
      ET_CHECK_OR_RETURN_ERROR(
          false,
          NotSupported,
          "Unsupported image encoder input dtype: %s",
          ::executorch::runtime::toString(expected_dtype));
    }

    auto expected_dims = input_meta.sizes();
    auto image_tensor_result =
        image.toTensor(/*with_batch*/ expected_dims.size() == 4);
    ET_CHECK_OK_OR_RETURN_ERROR(
        image_tensor_result.error(), "Failed to convert image to tensor");
    auto image_tensor = image_tensor_result.get();

    if (expected_dtype == ::executorch::aten::ScalarType::BFloat16) {
      auto image_tensor_return = convert_to_bfloat16(image_tensor);
      ET_CHECK_OK_OR_RETURN_ERROR(
          image_tensor_return.error(),
          "Failed to convert image tensor to bfloat16");
      image_tensor = image_tensor_return.get();
    }

    auto image_encoder_result =
        module_->execute(::executorch::extension::llm::kVisionEncoderMethod, image_tensor);
    ET_CHECK_OK_OR_RETURN_ERROR(image_encoder_result.error());
    auto image_encoder_outputs = image_encoder_result.get();

    auto encoder_output = ::executorch::extension::clone_tensor_ptr(
        image_encoder_outputs[0].toTensor());
    ET_LOG(
        Info,
        "Runtime prefill: image seq_len=%" PRId64 " start_pos=%" PRId64,
        encoder_output->size(1),
        start_pos);

    int64_t seq_len = encoder_output->size(1);
    if (seq_len == 0) {
      ET_LOG(Error, "The image encoder returned an empty output.");
      return ::executorch::runtime::Error::InvalidState;
    }
    std::vector<int64_t> cache_positions;
    auto cache_position_result = populate_start_pos_or_cache_position(
        module_,
        start_pos,
        cache_positions,
        static_cast<int>(seq_len),
        ::executorch::extension::llm::kTextModelMethod);
    ET_CHECK_OK_OR_RETURN_ERROR(cache_position_result.error());
    auto cache_position_tensor = cache_position_result.get();
    ET_LOG(
        Info,
        "Runtime prefill: text_decoder image seq_len=%" PRId64
        " start_pos=%" PRId64,
        seq_len,
        start_pos);

    auto prefill_result = module_->execute(
        ::executorch::extension::llm::kTextModelMethod,
        {encoder_output, cache_position_tensor});
    ET_CHECK_OK_OR_RETURN_ERROR(prefill_result.error());
    auto prefill_outputs = prefill_result.get();
    ET_CHECK_OR_RETURN_ERROR(
        !prefill_outputs.empty(),
        InvalidState,
        "Image encoder returned empty results when used to prefill decoder");

    start_pos += seq_len;
    return static_cast<uint64_t>(
        text_decoder_runner_->logits_to_token(prefill_outputs[0].toTensor()));
  }
};

TensorPtr make_zero_tensor(
    std::vector<executorch::aten::SizesType> sizes,
    executorch::aten::ScalarType type) {
  return ::executorch::extension::zeros(
      std::move(sizes),
      type,
      executorch::aten::TensorShapeDynamism::STATIC);
}

executorch::runtime::Result<TensorPtr> pad_embeddings_for_prewarm(
    const TensorPtr& src_embeddings,
    int64_t target_seq_len) {
  ET_CHECK_OR_RETURN_ERROR(
      src_embeddings->dim() == 3,
      InvalidArgument,
      "Expected rank-3 embeddings tensor for prewarm");
  ET_CHECK_OR_RETURN_ERROR(
      target_seq_len >= src_embeddings->size(1),
      InvalidArgument,
      "Prewarm target sequence length must be >= source sequence length");

  const auto scalar_type = src_embeddings->scalar_type();
  const auto hidden_size = src_embeddings->size(2);
  auto dst_embeddings = make_zero_tensor(
      {1,
       static_cast<executorch::aten::SizesType>(target_seq_len),
       static_cast<executorch::aten::SizesType>(hidden_size)},
      scalar_type);
  const size_t copy_elems = static_cast<size_t>(src_embeddings->numel());

  switch (scalar_type) {
    case executorch::aten::ScalarType::Float:
      std::copy(
          src_embeddings->const_data_ptr<float>(),
          src_embeddings->const_data_ptr<float>() + copy_elems,
          dst_embeddings->mutable_data_ptr<float>());
      break;
    case executorch::aten::ScalarType::Half:
      std::copy(
          src_embeddings->const_data_ptr<executorch::aten::Half>(),
          src_embeddings->const_data_ptr<executorch::aten::Half>() + copy_elems,
          dst_embeddings->mutable_data_ptr<executorch::aten::Half>());
      break;
    case executorch::aten::ScalarType::BFloat16:
      std::copy(
          src_embeddings->const_data_ptr<executorch::aten::BFloat16>(),
          src_embeddings->const_data_ptr<executorch::aten::BFloat16>() + copy_elems,
          dst_embeddings->mutable_data_ptr<executorch::aten::BFloat16>());
      break;
    default:
      ET_LOG(
          Error,
          "Unsupported embedding scalar type for prewarm padding: %s",
          ::executorch::runtime::toString(scalar_type));
      return ::executorch::runtime::Error::NotSupported;
  }

  return dst_embeddings;
}

executorch::runtime::Error prewarm_text_decoder_with_token_len(
    Module& module,
    int64_t token_len,
    int64_t start_pos) {
  auto token_input = make_zero_tensor(
      {1, static_cast<executorch::aten::SizesType>(token_len)},
      executorch::aten::ScalarType::Long);
  ET_LOG(Info, "Prewarm step: token_embedding(len=%" PRId64 ")", token_len);
  auto token_embedding_outputs =
      module.execute(::executorch::extension::llm::kTokenEmbeddingMethod, token_input);
  ET_CHECK_OK_OR_RETURN_ERROR(token_embedding_outputs.error());

  std::vector<int64_t> token_cache_positions;
  auto token_decode_pos_tensor_result = populate_start_pos_or_cache_position(
      &module,
      start_pos,
      token_cache_positions,
      static_cast<int>(token_len),
      ::executorch::extension::llm::kTextModelMethod);
  ET_CHECK_OK_OR_RETURN_ERROR(token_decode_pos_tensor_result.error());
  auto token_decode_pos_tensor = token_decode_pos_tensor_result.get();
  ET_LOG(
      Info,
      "Prewarm step: text_decoder(token embeddings len=%" PRId64
      " start_pos=%" PRId64 ")",
      token_len,
      start_pos);
  auto cloned_token_embeddings =
      ::executorch::extension::clone_tensor_ptr(token_embedding_outputs.get()[0].toTensor());
  auto token_decode_outputs = module.execute(
      ::executorch::extension::llm::kTextModelMethod,
      {cloned_token_embeddings, token_decode_pos_tensor});
  ET_CHECK_OK_OR_RETURN_ERROR(token_decode_outputs.error());
  return ::executorch::runtime::Error::Ok;
}

executorch::runtime::Error prewarm_split_text_decoder_with_token_len(
    Module& token_module,
    Module& text_module,
    int64_t token_len,
    int64_t start_pos) {
  auto token_input = make_zero_tensor(
      {1, static_cast<executorch::aten::SizesType>(token_len)},
      executorch::aten::ScalarType::Long);
  ET_LOG(
      Info,
      "Prewarm step(split): token_embedding(len=%" PRId64 ")",
      token_len);
  auto token_embedding_outputs = token_module.execute(
      ::executorch::extension::llm::kTokenEmbeddingMethod, token_input);
  ET_CHECK_OK_OR_RETURN_ERROR(token_embedding_outputs.error());

  std::vector<int64_t> token_cache_positions;
  auto token_decode_pos_tensor_result = populate_start_pos_or_cache_position(
      &text_module,
      start_pos,
      token_cache_positions,
      static_cast<int>(token_len),
      ::executorch::extension::llm::kTextModelMethod);
  ET_CHECK_OK_OR_RETURN_ERROR(token_decode_pos_tensor_result.error());
  auto token_decode_pos_tensor = token_decode_pos_tensor_result.get();

  ET_LOG(
      Info,
      "Prewarm step(split): text_decoder(token embeddings len=%" PRId64
      " start_pos=%" PRId64 ")",
      token_len,
      start_pos);
  auto cloned_token_embeddings =
      ::executorch::extension::clone_tensor_ptr(
          token_embedding_outputs.get()[0].toTensor());
  auto token_decode_outputs = text_module.execute(
      ::executorch::extension::llm::kTextModelMethod,
      {cloned_token_embeddings, token_decode_pos_tensor});
  ET_CHECK_OK_OR_RETURN_ERROR(token_decode_outputs.error());
  return ::executorch::runtime::Error::Ok;
}

executorch::runtime::Result<TensorPtr> run_vision_encoder_module(
    Module* vision_module,
    const MultimodalInput& image_input) {
  ET_CHECK_OR_RETURN_ERROR(
      image_input.is_image(),
      InvalidArgument,
      "Expected image input for vision encoder");
  const Image& image = image_input.get_image();

  auto vision_meta =
      vision_module->method_meta(::executorch::extension::llm::kVisionEncoderMethod);
  ET_CHECK_OK_OR_RETURN_ERROR(vision_meta.error());
  auto image_meta = vision_meta->input_tensor_meta(0);
  ET_CHECK_OK_OR_RETURN_ERROR(image_meta.error());
  auto expected_dtype = image_meta->scalar_type();

  auto image_tensor_result =
      image.toTensor(/*with_batch=*/image_meta->sizes().size() == 4);
  ET_CHECK_OK_OR_RETURN_ERROR(image_tensor_result.error());
  auto image_tensor = image_tensor_result.get();

  if (expected_dtype == ::executorch::aten::ScalarType::BFloat16) {
    auto image_tensor_bf16 = convert_to_bfloat16(image_tensor);
    ET_CHECK_OK_OR_RETURN_ERROR(image_tensor_bf16.error());
    image_tensor = image_tensor_bf16.get();
  } else {
    ET_CHECK_OR_RETURN_ERROR(
        expected_dtype == image_tensor->scalar_type(),
        InvalidArgument,
        "Unexpected vision encoder input dtype: got %s expected %s",
        ::executorch::runtime::toString(image_tensor->scalar_type()),
        ::executorch::runtime::toString(expected_dtype));
  }

  auto vision_outputs = vision_module->execute(
      ::executorch::extension::llm::kVisionEncoderMethod, image_tensor);
  ET_CHECK_OK_OR_RETURN_ERROR(vision_outputs.error());
  return ::executorch::extension::clone_tensor_ptr(
      vision_outputs.get()[0].toTensor());
}

executorch::runtime::Error prewarm_split_modules(
    Module& vision_module,
    Module& token_module,
    Module& text_module,
    int64_t max_context_len) {
  const int64_t prewarm_context_len =
      std::max<int64_t>(1, std::min<int64_t>(max_context_len, FLAGS_prewarm_context_len));
  ET_LOG(
      Info,
      "Prewarming split Vulkan modules with prewarm_context_len=%" PRId64
      " (model max_context_len=%" PRId64 ")",
      prewarm_context_len,
      max_context_len);

  const int64_t decode_start_pos = prewarm_context_len > 0 ? prewarm_context_len - 1 : 0;

  if (!FLAGS_skip_image_prewarm) {
    ET_CHECK_OK_OR_RETURN_ERROR(prewarm_split_text_decoder_with_token_len(
        token_module, text_module, prewarm_context_len, decode_start_pos));

    auto vision_meta =
        vision_module.method_meta(::executorch::extension::llm::kVisionEncoderMethod);
    ET_CHECK_OK_OR_RETURN_ERROR(vision_meta.error());
    auto image_meta = vision_meta->input_tensor_meta(0);
    ET_CHECK_OK_OR_RETURN_ERROR(image_meta.error());

    std::vector<executorch::aten::SizesType> image_sizes(
        image_meta->sizes().begin(), image_meta->sizes().end());
    auto image_input = make_zero_tensor(image_sizes, image_meta->scalar_type());
    ET_LOG(Info, "Prewarm step(split): vision_encoder");
    auto vision_outputs = vision_module.execute(
        ::executorch::extension::llm::kVisionEncoderMethod, image_input);
    ET_CHECK_OK_OR_RETURN_ERROR(vision_outputs.error());
    auto cloned_image_embeddings =
        ::executorch::extension::clone_tensor_ptr(vision_outputs.get()[0].toTensor());
    auto padded_image_embeddings_result =
        pad_embeddings_for_prewarm(cloned_image_embeddings, prewarm_context_len);
    ET_CHECK_OK_OR_RETURN_ERROR(padded_image_embeddings_result.error());
    auto padded_image_embeddings = padded_image_embeddings_result.get();

    int64_t image_start_pos = decode_start_pos;
    std::vector<int64_t> image_cache_positions;
    auto image_pos_tensor_result = populate_start_pos_or_cache_position(
        &text_module,
        image_start_pos,
        image_cache_positions,
        static_cast<int>(prewarm_context_len),
        ::executorch::extension::llm::kTextModelMethod);
    ET_CHECK_OK_OR_RETURN_ERROR(image_pos_tensor_result.error());
    auto image_pos_tensor = image_pos_tensor_result.get();
    ET_LOG(
        Info,
        "Prewarm step(split): text_decoder(image embeddings padded_len=%" PRId64
        " start_pos=%" PRId64 ")",
        prewarm_context_len,
        image_start_pos);
    auto image_decode_outputs = text_module.execute(
        ::executorch::extension::llm::kTextModelMethod,
        {padded_image_embeddings, image_pos_tensor});
    ET_CHECK_OK_OR_RETURN_ERROR(image_decode_outputs.error());
  } else {
    ET_LOG(Info, "Skipping split image-related prewarm by request");
  }

  ET_CHECK_OK_OR_RETURN_ERROR(prewarm_split_text_decoder_with_token_len(
      token_module, text_module, prewarm_context_len, decode_start_pos));
  ET_CHECK_OK_OR_RETURN_ERROR(
      prewarm_split_text_decoder_with_token_len(token_module, text_module, 5, 0));
  ET_CHECK_OK_OR_RETURN_ERROR(
      prewarm_split_text_decoder_with_token_len(token_module, text_module, 1, 0));

  ET_LOG(Info, "Split Vulkan prewarm finished");
  return ::executorch::runtime::Error::Ok;
}

executorch::runtime::Error prewarm_method_shapes(
    Module& module,
    int64_t max_context_len) {
  const int64_t prewarm_context_len =
      std::max<int64_t>(1, std::min<int64_t>(max_context_len, FLAGS_prewarm_context_len));
  ET_LOG(
      Info,
      "Prewarming Vulkan methods with prewarm_context_len=%" PRId64
      " (model max_context_len=%" PRId64 ")",
      prewarm_context_len,
      max_context_len);

  auto method_names_result = module.method_names();
  ET_CHECK_OK_OR_RETURN_ERROR(method_names_result.error());
  const std::unordered_set<std::string>& method_names = method_names_result.get();

  int64_t decode_start_pos = prewarm_context_len > 0 ? prewarm_context_len - 1 : 0;
  if (method_names.count(::executorch::extension::llm::kVisionEncoderMethod) > 0 &&
      !FLAGS_skip_image_prewarm) {
    ET_CHECK_OK_OR_RETURN_ERROR(prewarm_text_decoder_with_token_len(
        module, prewarm_context_len, decode_start_pos));

    auto vision_meta =
        module.method_meta(::executorch::extension::llm::kVisionEncoderMethod);
    ET_CHECK_OK_OR_RETURN_ERROR(vision_meta.error());
    auto image_meta = vision_meta->input_tensor_meta(0);
    ET_CHECK_OK_OR_RETURN_ERROR(image_meta.error());

    std::vector<executorch::aten::SizesType> image_sizes(
        image_meta->sizes().begin(), image_meta->sizes().end());
    auto image_input = make_zero_tensor(image_sizes, image_meta->scalar_type());
    ET_LOG(Info, "Prewarm step: vision_encoder");
    auto vision_outputs =
        module.execute(::executorch::extension::llm::kVisionEncoderMethod, image_input);
    ET_CHECK_OK_OR_RETURN_ERROR(vision_outputs.error());
    auto cloned_image_embeddings =
        ::executorch::extension::clone_tensor_ptr(vision_outputs.get()[0].toTensor());
    auto padded_image_embeddings_result =
        pad_embeddings_for_prewarm(cloned_image_embeddings, prewarm_context_len);
    ET_CHECK_OK_OR_RETURN_ERROR(padded_image_embeddings_result.error());
    auto padded_image_embeddings = padded_image_embeddings_result.get();

    int64_t image_start_pos = decode_start_pos;
    std::vector<int64_t> image_cache_positions;
    auto image_pos_tensor_result = populate_start_pos_or_cache_position(
        &module,
        image_start_pos,
        image_cache_positions,
        static_cast<int>(prewarm_context_len),
        ::executorch::extension::llm::kTextModelMethod);
    ET_CHECK_OK_OR_RETURN_ERROR(image_pos_tensor_result.error());
    auto image_pos_tensor = image_pos_tensor_result.get();
    ET_LOG(
        Info,
        "Prewarm step: text_decoder(image embeddings padded_len=%" PRId64
        " start_pos=%" PRId64 ")",
        prewarm_context_len,
        image_start_pos);
    auto image_decode_outputs = module.execute(
        ::executorch::extension::llm::kTextModelMethod,
        {padded_image_embeddings, image_pos_tensor});
    ET_CHECK_OK_OR_RETURN_ERROR(image_decode_outputs.error());
  } else {
    if (FLAGS_skip_image_prewarm &&
        method_names.count(::executorch::extension::llm::kVisionEncoderMethod) > 0) {
      ET_LOG(Info, "Skipping image-related prewarm by request");
    }
  }

  ET_CHECK_OK_OR_RETURN_ERROR(
      prewarm_text_decoder_with_token_len(module, prewarm_context_len, decode_start_pos));
  ET_CHECK_OK_OR_RETURN_ERROR(prewarm_text_decoder_with_token_len(module, 5, 0));
  ET_CHECK_OK_OR_RETURN_ERROR(prewarm_text_decoder_with_token_len(module, 1, 0));

  ET_LOG(Info, "Vulkan prewarm finished");
  return ::executorch::runtime::Error::Ok;
}

executorch::runtime::Error run_vision_encoder_once(const std::string& model_path) {
  auto module =
      std::make_unique<Module>(model_path, Module::LoadMode::MmapUseMlockIgnoreErrors);

  auto vision_meta = module->method_meta(::executorch::extension::llm::kVisionEncoderMethod);
  ET_CHECK_OK_OR_RETURN_ERROR(vision_meta.error());
  auto image_meta = vision_meta->input_tensor_meta(0);
  ET_CHECK_OK_OR_RETURN_ERROR(image_meta.error());

  std::vector<executorch::aten::SizesType> image_sizes(
      image_meta->sizes().begin(), image_meta->sizes().end());
  auto image_input = make_zero_tensor(image_sizes, image_meta->scalar_type());

  ET_LOG(Info, "Diagnostic: running vision_encoder only");
  const auto start = std::chrono::steady_clock::now();
  auto vision_outputs =
      module->execute(::executorch::extension::llm::kVisionEncoderMethod, image_input);
  const auto end = std::chrono::steady_clock::now();
  ET_CHECK_OK_OR_RETURN_ERROR(vision_outputs.error());

  const auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  ET_LOG(
      Info,
      "Diagnostic: vision_encoder finished in %lld ms with %zu outputs",
      static_cast<long long>(elapsed_ms),
      vision_outputs.get().size());
  return ::executorch::runtime::Error::Ok;
}

executorch::runtime::Error run_text_decoder_once(const std::string& model_path) {
  auto module =
      std::make_unique<Module>(model_path, Module::LoadMode::MmapUseMlockIgnoreErrors);

  const int64_t seq_len = std::max<int64_t>(1, FLAGS_decoder_seq_len);
  const int64_t start_pos = std::max<int64_t>(0, FLAGS_decoder_start_pos);

  TensorPtr embeddings;
  if (FLAGS_decoder_use_token_embedding) {
    auto token_input = make_zero_tensor(
        {1, static_cast<executorch::aten::SizesType>(seq_len)},
        executorch::aten::ScalarType::Long);
    ET_LOG(
        Info,
        "Diagnostic: running token_embedding for text_decoder-only path (len=%" PRId64 ")",
        seq_len);
    auto token_embedding_outputs =
        module->execute(::executorch::extension::llm::kTokenEmbeddingMethod, token_input);
    ET_CHECK_OK_OR_RETURN_ERROR(token_embedding_outputs.error());
    embeddings = ::executorch::extension::clone_tensor_ptr(
        token_embedding_outputs.get()[0].toTensor());
  } else {
    auto text_meta = module->method_meta(::executorch::extension::llm::kTextModelMethod);
    ET_CHECK_OK_OR_RETURN_ERROR(text_meta.error());
    auto embedding_meta = text_meta->input_tensor_meta(0);
    ET_CHECK_OK_OR_RETURN_ERROR(embedding_meta.error());
    auto embedding_sizes = embedding_meta->sizes();
    ET_CHECK_OR_RETURN_ERROR(
        embedding_sizes.size() == 3,
        InvalidArgument,
        "text_decoder embedding input must be rank-3");
    embeddings = make_zero_tensor(
        {
            static_cast<executorch::aten::SizesType>(embedding_sizes[0]),
            static_cast<executorch::aten::SizesType>(seq_len),
            static_cast<executorch::aten::SizesType>(embedding_sizes[2]),
        },
        embedding_meta->scalar_type());
  }

  int64_t mutable_start_pos = start_pos;
  std::vector<int64_t> cache_positions;
  auto cache_position_result = populate_start_pos_or_cache_position(
      module.get(),
      mutable_start_pos,
      cache_positions,
      static_cast<int>(seq_len),
      ::executorch::extension::llm::kTextModelMethod);
  ET_CHECK_OK_OR_RETURN_ERROR(cache_position_result.error());
  auto cache_position_tensor = cache_position_result.get();

  ET_LOG(
      Info,
      "Diagnostic: running text_decoder only (seq_len=%" PRId64
      ", start_pos=%" PRId64 ", from_token_embedding=%d)",
      seq_len,
      start_pos,
      FLAGS_decoder_use_token_embedding ? 1 : 0);
  const auto start = std::chrono::steady_clock::now();
  auto outputs = module->execute(
      ::executorch::extension::llm::kTextModelMethod,
      {embeddings, cache_position_tensor});
  const auto end = std::chrono::steady_clock::now();
  ET_CHECK_OK_OR_RETURN_ERROR(outputs.error());

  const auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  ET_LOG(
      Info,
      "Diagnostic: text_decoder finished in %lld ms with %zu outputs",
      static_cast<long long>(elapsed_ms),
      outputs.get().size());
  return ::executorch::runtime::Error::Ok;
}

executorch::runtime::Result<uint64_t> prefill_text_input(
    Module* token_module,
    Module* text_module,
    TextDecoderRunner* text_decoder_runner,
    ::tokenizers::Tokenizer* tokenizer,
    const std::string& text,
    int64_t& start_pos) {
  auto encode_result = tokenizer->encode(text, 0, 0);
  if (!encode_result.ok()) {
    ET_LOG(
        Error,
        "Tokenizers error code %d",
        static_cast<uint32_t>(encode_result.error()));
    return ::executorch::runtime::Error::InvalidArgument;
  }
  auto tokens = std::move(*encode_result);
  auto text_tensor = ::executorch::extension::from_blob(
      tokens.data(),
      {1, static_cast<executorch::aten::SizesType>(tokens.size())},
      ::executorch::aten::ScalarType::Long);
  ET_LOG(
      Info,
      "Runtime prefill(custom): text len=%zu start_pos=%" PRId64,
      tokens.size(),
      start_pos);

  auto token_embedding_result = token_module->execute(
      ::executorch::extension::llm::kTokenEmbeddingMethod, text_tensor);
  ET_CHECK_OK_OR_RETURN_ERROR(token_embedding_result.error());
  auto token_embedding_outputs = token_embedding_result.get();
  auto embeddings = ::executorch::extension::clone_tensor_ptr(
      token_embedding_outputs[0].toTensor());

  std::vector<int64_t> cache_positions;
  auto cache_position_result = populate_start_pos_or_cache_position(
      text_module,
      start_pos,
      cache_positions,
      static_cast<int>(embeddings->size(1)),
      ::executorch::extension::llm::kTextModelMethod);
  ET_CHECK_OK_OR_RETURN_ERROR(cache_position_result.error());
  auto cache_position_tensor = cache_position_result.get();

  auto outputs = text_module->execute(
      ::executorch::extension::llm::kTextModelMethod,
      {embeddings, cache_position_tensor});
  ET_CHECK_OK_OR_RETURN_ERROR(outputs.error());

  start_pos += embeddings->size(1);
  return static_cast<uint64_t>(
      text_decoder_runner->logits_to_token(outputs.get()[0].toTensor()));
}

executorch::runtime::Result<size_t> get_tokenized_length(
    ::tokenizers::Tokenizer* tokenizer,
    const std::string& text) {
  auto encode_result = tokenizer->encode(text, 0, 0);
  if (!encode_result.ok()) {
    ET_LOG(
        Error,
        "Tokenizers error code %d",
        static_cast<uint32_t>(encode_result.error()));
    return ::executorch::runtime::Error::InvalidArgument;
  }
  return encode_result->size();
}

void print_generation_metrics(const Stats& stats) {
  ::executorch::extension::llm::print_report(stats);

  const double model_load_time_ms =
      static_cast<double>(stats.model_load_end_ms - stats.model_load_start_ms);
  const double total_inference_time_ms =
      static_cast<double>(stats.inference_end_ms - stats.inference_start_ms);
  const double ttft_ms =
      static_cast<double>(stats.first_token_ms - stats.inference_start_ms);
  const double prefill_time_ms =
      static_cast<double>(stats.prompt_eval_end_ms - stats.inference_start_ms);
  const double generation_time_ms =
      static_cast<double>(stats.inference_end_ms - stats.prompt_eval_end_ms);
  const double total_throughput_tokens_per_second =
      total_inference_time_ms > 0.0
      ? static_cast<double>(stats.num_generated_tokens) * 1000.0 /
          total_inference_time_ms
      : 0.0;
  const double prefill_tokens_per_second =
      prefill_time_ms > 0.0
      ? static_cast<double>(stats.num_prompt_tokens) * 1000.0 / prefill_time_ms
      : 0.0;
  const double avg_time_per_generated_token_ms =
      stats.num_generated_tokens > 0
      ? generation_time_ms / static_cast<double>(stats.num_generated_tokens)
      : 0.0;
  const double decode_tokens_per_second =
      generation_time_ms > 0.0
      ? static_cast<double>(stats.num_generated_tokens) * 1000.0 /
          generation_time_ms
      : 0.0;

  ET_LOG(
      Info,
      "Metrics summary:");
  ET_LOG(
      Info,
      "  model_load_time_ms=%0.3f",
      model_load_time_ms);
  ET_LOG(
      Info,
      "  total_inference_time_ms=%0.3f",
      total_inference_time_ms);
  ET_LOG(
      Info,
      "  ttft_ms=%0.3f",
      ttft_ms);
  ET_LOG(
      Info,
      "  multimodal_prefill_time_ms=%0.3f",
      prefill_time_ms);
  ET_LOG(
      Info,
      "  multimodal_prefill_speed_tokens_per_s=%0.3f",
      prefill_tokens_per_second);
  ET_LOG(
      Info,
      "  decode_time_ms=%0.3f",
      generation_time_ms);
  ET_LOG(
      Info,
      "  decode_speed_tokens_per_s=%0.3f",
      decode_tokens_per_second);
  ET_LOG(
      Info,
      "  avg_decode_time_per_token_ms=%0.3f",
      avg_time_per_generated_token_ms);
  ET_LOG(
      Info,
      "  total_generated_token_throughput_tokens_per_s=%0.3f",
      total_throughput_tokens_per_second);
  ET_LOG(
      Info,
      "  prompt_tokens=%" PRId64,
      stats.num_prompt_tokens);
  ET_LOG(
      Info,
      "  generated_tokens=%" PRId64,
      stats.num_generated_tokens);
  ET_LOG(
      Info,
      "  sampling_time_ms=%ld",
      stats.aggregate_sampling_time_ms);
}

executorch::runtime::Result<uint64_t> prefill_image_embeddings(
    Module* text_module,
    TextDecoderRunner* text_decoder_runner,
    const TensorPtr& image_embeddings,
    int64_t& start_pos) {
  ET_LOG(
      Info,
      "Runtime prefill(custom): image_embeddings seq_len=%" PRId64
      " start_pos=%" PRId64,
      image_embeddings->size(1),
      start_pos);
  auto embeddings = ::executorch::extension::clone_tensor_ptr(image_embeddings);
  std::vector<int64_t> cache_positions;
  auto cache_position_result = populate_start_pos_or_cache_position(
      text_module,
      start_pos,
      cache_positions,
      static_cast<int>(embeddings->size(1)),
      ::executorch::extension::llm::kTextModelMethod);
  ET_CHECK_OK_OR_RETURN_ERROR(cache_position_result.error());
  auto cache_position_tensor = cache_position_result.get();

  auto outputs = text_module->execute(
      ::executorch::extension::llm::kTextModelMethod,
      {embeddings, cache_position_tensor});
  ET_CHECK_OK_OR_RETURN_ERROR(outputs.error());

  start_pos += embeddings->size(1);
  return static_cast<uint64_t>(
      text_decoder_runner->logits_to_token(outputs.get()[0].toTensor()));
}

executorch::runtime::Error run_with_precomputed_image_embeddings(
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer) {
  const std::string prefix_text =
      "<|im_start|>User:<fake_token_around_image><global-img>";
  const std::string suffix_text =
      std::string("<fake_token_around_image>") + FLAGS_prompt +
      "<end_of_utterance>\nAssistant:";
  auto stats = std::make_unique<Stats>();
  stats->reset(/*all_stats=*/true);
  stats->model_load_start_ms = ::executorch::extension::llm::time_in_ms();
  auto module = std::make_unique<Module>(
      FLAGS_model_path, Module::LoadMode::MmapUseMlockIgnoreErrors);

  ET_CHECK_OK_OR_RETURN_ERROR(
      module->load_method(::executorch::extension::llm::kTokenEmbeddingMethod));
  ET_CHECK_OK_OR_RETURN_ERROR(
      module->load_method(::executorch::extension::llm::kTextModelMethod));

  auto metadata_result =
      ::executorch::extension::llm::get_llm_metadata(tokenizer.get(), module.get());
  ET_CHECK_OK_OR_RETURN_ERROR(metadata_result.error());
  auto metadata = metadata_result.get();
  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>(
      ::executorch::extension::llm::get_eos_ids(tokenizer.get(), module.get()));

  auto io_manager = std::make_unique<IOManager>(*module);
  auto text_decoder_runner =
      std::make_unique<MultimodalDecoderRunner>(module.get(), io_manager.get());
  auto text_token_generator = std::make_unique<TextTokenGenerator>(
      tokenizer.get(),
      text_decoder_runner.get(),
      metadata.at(::executorch::extension::llm::kUseKVCache),
      std::move(eos_ids),
      stats.get());
  ET_CHECK_OK_OR_RETURN_ERROR(text_token_generator->load());
  stats->model_load_end_ms = ::executorch::extension::llm::time_in_ms();

  int64_t start_pos = 0;
  auto image_embeddings =
      load_embeddings_bin(FLAGS_image_embeddings_bin_path, "image embeddings");
  auto prefix_len_result = get_tokenized_length(tokenizer.get(), prefix_text);
  ET_CHECK_OK_OR_RETURN_ERROR(prefix_len_result.error());
  auto suffix_len_result = get_tokenized_length(tokenizer.get(), suffix_text);
  ET_CHECK_OK_OR_RETURN_ERROR(suffix_len_result.error());

  if (!FLAGS_skip_prewarm) {
    const int64_t prefix_len = static_cast<int64_t>(prefix_len_result.get());
    const int64_t image_len = image_embeddings->size(1);
    const int64_t suffix_len = static_cast<int64_t>(suffix_len_result.get());
    const int64_t image_start_pos = prefix_len;
    const int64_t suffix_start_pos = prefix_len + image_len;
    const int64_t decode_start_pos = suffix_start_pos + suffix_len - 1;
    const int64_t max_context_len =
        metadata.at(::executorch::extension::llm::kMaxContextLen);
    const int64_t warm_span_len =
        std::max<int64_t>(
            1, std::min<int64_t>(max_context_len, FLAGS_prewarm_context_len));
    const int64_t warm_span_start = std::max<int64_t>(0, max_context_len - warm_span_len);
    ET_LOG(
        Info,
        "Custom prewarm lengths: prefix=%" PRId64 " image=%" PRId64
        " suffix=%" PRId64 " wide_span=%" PRId64 "@%" PRId64,
        prefix_len,
        image_len,
        suffix_len,
        warm_span_len,
        warm_span_start);
    ET_CHECK_OK_OR_RETURN_ERROR(
        prewarm_text_decoder_with_token_len(*module, warm_span_len, warm_span_start));
    ET_CHECK_OK_OR_RETURN_ERROR(prewarm_text_decoder_with_token_len(
        *module, suffix_len, suffix_start_pos));
    ET_CHECK_OK_OR_RETURN_ERROR(prewarm_text_decoder_with_token_len(
        *module, image_len, image_start_pos));
    ET_CHECK_OK_OR_RETURN_ERROR(
        prewarm_text_decoder_with_token_len(*module, prefix_len, 0));
    ET_CHECK_OK_OR_RETURN_ERROR(
        prewarm_text_decoder_with_token_len(*module, 1, decode_start_pos));
  } else {
    ET_LOG(Info, "Skipping custom prewarm by request");
  }

  stats->inference_start_ms = ::executorch::extension::llm::time_in_ms();
  auto prefix_result = prefill_text_input(
      module.get(),
      module.get(),
      text_decoder_runner.get(),
      tokenizer.get(),
      prefix_text,
      start_pos);
  ET_CHECK_OK_OR_RETURN_ERROR(prefix_result.error());

  auto image_result = prefill_image_embeddings(
      module.get(), text_decoder_runner.get(), image_embeddings, start_pos);
  ET_CHECK_OK_OR_RETURN_ERROR(image_result.error());

  auto suffix_result = prefill_text_input(
      module.get(),
      module.get(),
      text_decoder_runner.get(),
      tokenizer.get(),
      suffix_text,
      start_pos);
  ET_CHECK_OK_OR_RETURN_ERROR(suffix_result.error());

  stats->prompt_eval_end_ms = ::executorch::extension::llm::time_in_ms();
  stats->first_token_ms = stats->prompt_eval_end_ms;
  stats->num_prompt_tokens = start_pos;

  uint64_t cur_token = suffix_result.get();
  auto decode_result = tokenizer->decode(cur_token, cur_token);
  if (!decode_result.ok()) {
    ET_LOG(
        Error,
        "Tokenizers error code %d",
        static_cast<uint32_t>(decode_result.error()));
    return ::executorch::runtime::Error::InvalidArgument;
  }
  ::executorch::extension::llm::safe_printf(decode_result->c_str());
  fflush(stdout);

  text_token_generator->set_ignore_eos(false);
  auto generate_result = text_token_generator->generate(
      {cur_token},
      start_pos,
      FLAGS_max_new_tokens - 1,
      static_cast<float>(FLAGS_temperature),
      [](const std::string& piece) {
        ::executorch::extension::llm::safe_printf(piece.c_str());
        fflush(stdout);
      });
  ET_CHECK_OK_OR_RETURN_ERROR(generate_result.error());
  stats->num_generated_tokens = generate_result.get();
  stats->inference_end_ms = ::executorch::extension::llm::time_in_ms();
  printf("\n");
  print_generation_metrics(*stats);
  return ::executorch::runtime::Error::Ok;
}

executorch::runtime::Error run_with_precomputed_prompt_embeddings(
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer) {
  auto stats = std::make_unique<Stats>();
  stats->reset(/*all_stats=*/true);
  stats->model_load_start_ms = ::executorch::extension::llm::time_in_ms();
  auto module = std::make_unique<Module>(
      FLAGS_model_path, Module::LoadMode::MmapUseMlockIgnoreErrors);

  ET_CHECK_OK_OR_RETURN_ERROR(
      module->load_method(::executorch::extension::llm::kTokenEmbeddingMethod));
  ET_CHECK_OK_OR_RETURN_ERROR(
      module->load_method(::executorch::extension::llm::kTextModelMethod));

  auto metadata_result =
      ::executorch::extension::llm::get_llm_metadata(tokenizer.get(), module.get());
  ET_CHECK_OK_OR_RETURN_ERROR(metadata_result.error());
  auto metadata = metadata_result.get();
  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>(
      ::executorch::extension::llm::get_eos_ids(tokenizer.get(), module.get()));

  auto io_manager = std::make_unique<IOManager>(*module);
  auto text_decoder_runner =
      std::make_unique<MultimodalDecoderRunner>(module.get(), io_manager.get());
  auto text_token_generator = std::make_unique<TextTokenGenerator>(
      tokenizer.get(),
      text_decoder_runner.get(),
      metadata.at(::executorch::extension::llm::kUseKVCache),
      std::move(eos_ids),
      stats.get());
  ET_CHECK_OK_OR_RETURN_ERROR(text_token_generator->load());
  stats->model_load_end_ms = ::executorch::extension::llm::time_in_ms();

  auto prompt_embeddings =
      load_embeddings_bin(FLAGS_prompt_embeddings_bin_path, "prompt embeddings");
  const int64_t prompt_len = prompt_embeddings->size(1);
  const int64_t decode_start_pos = prompt_len;

  if (!FLAGS_skip_prewarm) {
    const int64_t max_context_len =
        metadata.at(::executorch::extension::llm::kMaxContextLen);
    ET_LOG(
        Info,
        "Prompt-embedding prewarm lengths: prompt=%" PRId64
        " decode=1@%" PRId64 " max_decode=1@%" PRId64,
        prompt_len,
        decode_start_pos,
        max_context_len - 1);
    ET_CHECK_OK_OR_RETURN_ERROR(
        prewarm_text_decoder_with_token_len(*module, 1, max_context_len - 1));
    ET_CHECK_OK_OR_RETURN_ERROR(
        prewarm_text_decoder_with_token_len(*module, prompt_len, 0));
    ET_CHECK_OK_OR_RETURN_ERROR(
        prewarm_text_decoder_with_token_len(*module, 1, decode_start_pos));
  } else {
    ET_LOG(Info, "Skipping prompt-embedding prewarm by request");
  }

  int64_t start_pos = 0;
  stats->inference_start_ms = ::executorch::extension::llm::time_in_ms();
  auto first_token_result = prefill_image_embeddings(
      module.get(), text_decoder_runner.get(), prompt_embeddings, start_pos);
  ET_CHECK_OK_OR_RETURN_ERROR(first_token_result.error());

  stats->prompt_eval_end_ms = ::executorch::extension::llm::time_in_ms();
  stats->first_token_ms = stats->prompt_eval_end_ms;
  stats->num_prompt_tokens = start_pos;

  uint64_t cur_token = first_token_result.get();
  auto decode_result = tokenizer->decode(cur_token, cur_token);
  if (!decode_result.ok()) {
    ET_LOG(
        Error,
        "Tokenizers error code %d",
        static_cast<uint32_t>(decode_result.error()));
    return ::executorch::runtime::Error::InvalidArgument;
  }
  ::executorch::extension::llm::safe_printf(decode_result->c_str());
  fflush(stdout);

  text_token_generator->set_ignore_eos(false);
  auto generate_result = text_token_generator->generate(
      {cur_token},
      start_pos,
      FLAGS_max_new_tokens - 1,
      static_cast<float>(FLAGS_temperature),
      [](const std::string& piece) {
        ::executorch::extension::llm::safe_printf(piece.c_str());
        fflush(stdout);
      });
  ET_CHECK_OK_OR_RETURN_ERROR(generate_result.error());
  stats->num_generated_tokens = generate_result.get();
  stats->inference_end_ms = ::executorch::extension::llm::time_in_ms();
  printf("\n");
  print_generation_metrics(*stats);
  return ::executorch::runtime::Error::Ok;
}

executorch::runtime::Error run_with_split_vulkan_modules(
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer) {
  const std::string prefix_text =
      "<|im_start|>User:<fake_token_around_image><global-img>";
  const std::string suffix_text =
      std::string("<fake_token_around_image>") + FLAGS_prompt +
      "<end_of_utterance>\nAssistant:";
  auto stats = std::make_unique<Stats>();
  stats->reset(/*all_stats=*/true);
  stats->model_load_start_ms = ::executorch::extension::llm::time_in_ms();

  auto vision_module = std::make_unique<Module>(
      FLAGS_vision_model_path, Module::LoadMode::MmapUseMlockIgnoreErrors);
  auto token_module = std::make_unique<Module>(
      FLAGS_token_model_path, Module::LoadMode::MmapUseMlockIgnoreErrors);
  auto text_module = std::make_unique<Module>(
      FLAGS_text_model_path, Module::LoadMode::MmapUseMlockIgnoreErrors);

  ET_CHECK_OK_OR_RETURN_ERROR(
      vision_module->load_method(::executorch::extension::llm::kVisionEncoderMethod));

  auto metadata_result = ::executorch::extension::llm::get_llm_metadata(
      tokenizer.get(), text_module.get());
  ET_CHECK_OK_OR_RETURN_ERROR(metadata_result.error());
  auto metadata = metadata_result.get();
  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>(
      ::executorch::extension::llm::get_eos_ids(tokenizer.get(), text_module.get()));

  auto text_decoder_runner = std::make_unique<SplitTextDecoderRunner>(
      token_module.get(), text_module.get());
  auto text_token_generator = std::make_unique<TextTokenGenerator>(
      tokenizer.get(),
      text_decoder_runner.get(),
      metadata.at(::executorch::extension::llm::kUseKVCache),
      std::move(eos_ids),
      stats.get());
  ET_CHECK_OK_OR_RETURN_ERROR(text_token_generator->load());
  stats->model_load_end_ms = ::executorch::extension::llm::time_in_ms();

  auto image_input = load_image_bin(FLAGS_image_bin_path);
  auto prefix_len_result = get_tokenized_length(tokenizer.get(), prefix_text);
  ET_CHECK_OK_OR_RETURN_ERROR(prefix_len_result.error());
  auto suffix_len_result = get_tokenized_length(tokenizer.get(), suffix_text);
  ET_CHECK_OK_OR_RETURN_ERROR(suffix_len_result.error());

  if (!FLAGS_skip_prewarm) {
    ET_CHECK_OK_OR_RETURN_ERROR(prewarm_split_modules(
        *vision_module,
        *token_module,
        *text_module,
        metadata.at(::executorch::extension::llm::kMaxContextLen)));
  } else {
    ET_LOG(Info, "Skipping split Vulkan prewarm by request");
  }

  stats->inference_start_ms = ::executorch::extension::llm::time_in_ms();
  int64_t start_pos = 0;
  auto prefix_result = prefill_text_input(
      token_module.get(),
      text_module.get(),
      text_decoder_runner.get(),
      tokenizer.get(),
      prefix_text,
      start_pos);
  ET_CHECK_OK_OR_RETURN_ERROR(prefix_result.error());

  auto image_embeddings_result =
      run_vision_encoder_module(vision_module.get(), image_input);
  ET_CHECK_OK_OR_RETURN_ERROR(image_embeddings_result.error());
  auto image_embeddings = image_embeddings_result.get();

  auto image_result = prefill_image_embeddings(
      text_module.get(),
      text_decoder_runner.get(),
      image_embeddings,
      start_pos);
  ET_CHECK_OK_OR_RETURN_ERROR(image_result.error());

  auto suffix_result = prefill_text_input(
      token_module.get(),
      text_module.get(),
      text_decoder_runner.get(),
      tokenizer.get(),
      suffix_text,
      start_pos);
  ET_CHECK_OK_OR_RETURN_ERROR(suffix_result.error());

  stats->prompt_eval_end_ms = ::executorch::extension::llm::time_in_ms();
  stats->first_token_ms = stats->prompt_eval_end_ms;
  stats->num_prompt_tokens = start_pos;

  uint64_t cur_token = suffix_result.get();
  auto decode_result = tokenizer->decode(cur_token, cur_token);
  if (!decode_result.ok()) {
    ET_LOG(
        Error,
        "Tokenizers error code %d",
        static_cast<uint32_t>(decode_result.error()));
    return ::executorch::runtime::Error::InvalidArgument;
  }
  ::executorch::extension::llm::safe_printf(decode_result->c_str());
  fflush(stdout);

  text_token_generator->set_ignore_eos(false);
  auto generate_result = text_token_generator->generate(
      {cur_token},
      start_pos,
      FLAGS_max_new_tokens - 1,
      static_cast<float>(FLAGS_temperature),
      [](const std::string& piece) {
        ::executorch::extension::llm::safe_printf(piece.c_str());
        fflush(stdout);
      });
  ET_CHECK_OK_OR_RETURN_ERROR(generate_result.error());
  stats->num_generated_tokens = generate_result.get();
  stats->inference_end_ms = ::executorch::extension::llm::time_in_ms();
  printf("\n");
  print_generation_metrics(*stats);
  return ::executorch::runtime::Error::Ok;
}

std::unique_ptr<MultimodalRunner> create_prewarmed_multimodal_runner(
    const std::string& model_path,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer) {
  if (!tokenizer || !tokenizer->is_loaded()) {
    ET_LOG(Error, "Tokenizer is null or not loaded");
    return nullptr;
  }

  auto module = std::make_unique<Module>(
      model_path, Module::LoadMode::MmapUseMlockIgnoreErrors);

  auto metadata_result =
      ::executorch::extension::llm::get_llm_metadata(tokenizer.get(), module.get());
  if (!metadata_result.ok()) {
    ET_LOG(Error, "Failed to read LLM metadata");
    return nullptr;
  }
  auto metadata = metadata_result.get();

  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>(
      ::executorch::extension::llm::get_eos_ids(tokenizer.get(), module.get()));

  auto io_manager = std::make_unique<IOManager>(*module);
  auto text_decoder_runner =
      std::make_unique<MultimodalDecoderRunner>(module.get(), io_manager.get());
  auto multimodal_prefiller = std::make_unique<CloningMultimodalPrefiller>(
      module.get(),
      text_decoder_runner.get(),
      tokenizer.get(),
      io_manager.get());
  auto stats = std::make_unique<Stats>();
  auto text_token_generator = std::make_unique<TextTokenGenerator>(
      tokenizer.get(),
      text_decoder_runner.get(),
      metadata.at(::executorch::extension::llm::kUseKVCache),
      std::move(eos_ids),
      stats.get());

  auto load_error = multimodal_prefiller->load();
  if (load_error != ::executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to load multimodal prefiller");
    return nullptr;
  }
  load_error = text_token_generator->load();
  if (load_error != ::executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to load text token generator");
    return nullptr;
  }
  if (!FLAGS_skip_prewarm) {
    load_error = prewarm_method_shapes(
        *module, metadata.at(::executorch::extension::llm::kMaxContextLen));
    if (load_error != ::executorch::runtime::Error::Ok) {
      ET_LOG(Error, "Failed to prewarm Vulkan method shapes");
      return nullptr;
    }
  } else {
    ET_LOG(Info, "Skipping Vulkan prewarm by request");
  }

  return std::make_unique<MultimodalRunner>(
      std::move(metadata),
      std::move(tokenizer),
      std::move(module),
      std::move(text_decoder_runner),
      std::move(multimodal_prefiller),
      std::move(io_manager),
      std::move(text_token_generator),
      std::move(stats));
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

#if defined(ET_USE_THREADPOOL)
  uint32_t num_threads = FLAGS_cpu_threads == -1
      ? ::executorch::extension::cpuinfo::get_num_performant_cores()
      : static_cast<uint32_t>(FLAGS_cpu_threads);
  if (num_threads > 0) {
    ::executorch::extension::threadpool::get_threadpool()
        ->_unsafe_reset_threadpool(num_threads);
  }
#endif

  if (FLAGS_run_vision_only) {
    auto error = run_vision_encoder_once(FLAGS_model_path);
    return error == ::executorch::runtime::Error::Ok ? 0 : 1;
  }
  if (FLAGS_run_text_decoder_only) {
    auto error = run_text_decoder_once(FLAGS_model_path);
    return error == ::executorch::runtime::Error::Ok ? 0 : 1;
  }

  auto tokenizer =
      ::executorch::extension::llm::load_tokenizer(FLAGS_tokenizer_path);
  if (tokenizer == nullptr) {
    ET_LOG(Error, "Failed to load tokenizer: %s", FLAGS_tokenizer_path.c_str());
    return 1;
  }

  const bool any_split_path = !FLAGS_vision_model_path.empty() ||
      !FLAGS_token_model_path.empty() || !FLAGS_text_model_path.empty();
  const bool all_split_paths = !FLAGS_vision_model_path.empty() &&
      !FLAGS_token_model_path.empty() && !FLAGS_text_model_path.empty();
  if (any_split_path && !all_split_paths) {
    ET_LOG(
        Error,
        "Split-module execution requires --vision_model_path, --token_model_path, and --text_model_path");
    return 1;
  }
  if (all_split_paths) {
    auto error = run_with_split_vulkan_modules(std::move(tokenizer));
    return error == ::executorch::runtime::Error::Ok ? 0 : 1;
  }

  if (!FLAGS_prompt_embeddings_bin_path.empty()) {
    auto error = run_with_precomputed_prompt_embeddings(std::move(tokenizer));
    return error == ::executorch::runtime::Error::Ok ? 0 : 1;
  }

  if (!FLAGS_image_embeddings_bin_path.empty()) {
    auto error = run_with_precomputed_image_embeddings(std::move(tokenizer));
    return error == ::executorch::runtime::Error::Ok ? 0 : 1;
  }

  auto runner = create_prewarmed_multimodal_runner(
      FLAGS_model_path, std::move(tokenizer));
  if (runner == nullptr) {
    ET_LOG(Error, "Failed to create multimodal runner");
    return 1;
  }

  auto load_error = runner->load();
  if (load_error != ::executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to load multimodal runner");
    return 1;
  }

  std::vector<MultimodalInput> inputs = {
      make_text_input("<|im_start|>User:<fake_token_around_image><global-img>"),
      load_image_bin(FLAGS_image_bin_path),
      make_text_input(
          std::string("<fake_token_around_image>") + FLAGS_prompt +
          "<end_of_utterance>\nAssistant:"),
  };

  ::executorch::extension::llm::GenerationConfig config;
  config.max_new_tokens = FLAGS_max_new_tokens;
  config.temperature = static_cast<float>(FLAGS_temperature);

  if (FLAGS_warmup) {
    auto warmup_error = runner->generate(inputs, config);
    if (warmup_error != ::executorch::runtime::Error::Ok) {
      ET_LOG(Error, "Warmup failed");
      return 1;
    }
    runner->reset();
  }

  auto error = runner->generate(inputs, config);
  if (error != ::executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Generation failed");
    return 1;
  }

  return 0;
}
