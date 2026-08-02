// Minimal llama.cpp callback capture for a fixed prompt. It exports genuine
// per-block l_out states rather than terminal embeddings or logits.
#include <windows.h>

#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct Api {
    HMODULE llama = nullptr;
    HMODULE ggml_base = nullptr;
    HMODULE ggml_loader = nullptr;

    decltype(&llama_backend_init) backend_init = nullptr;
    decltype(&llama_backend_free) backend_free = nullptr;
    decltype(&llama_model_default_params) model_default_params = nullptr;
    decltype(&llama_context_default_params) context_default_params = nullptr;
    decltype(&llama_model_load_from_file) model_load_from_file = nullptr;
    decltype(&llama_model_free) model_free = nullptr;
    decltype(&llama_model_get_vocab) model_get_vocab = nullptr;
    decltype(&llama_model_n_layer) model_n_layer = nullptr;
    decltype(&llama_model_n_embd) model_n_embd = nullptr;
    decltype(&llama_init_from_model) init_from_model = nullptr;
    decltype(&llama_free) context_free = nullptr;
    decltype(&llama_tokenize) tokenize = nullptr;
    decltype(&llama_batch_get_one) batch_get_one = nullptr;
    decltype(&llama_decode) decode = nullptr;
    decltype(&ggml_backend_tensor_get) tensor_get = nullptr;
    decltype(&ggml_backend_load_all_from_path) load_backends_from_path = nullptr;
};

template <typename T>
static bool load_symbol(HMODULE module, const char * name, T & target) {
    target = reinterpret_cast<T>(GetProcAddress(module, name));
    if (target == nullptr) {
        std::cerr << "Missing symbol: " << name << "\n";
        return false;
    }
    return true;
}

static bool initialise_api(const std::string & runtime_dir, Api & api) {
    const std::string llama_path = runtime_dir + "\\llama.dll";
    const std::string ggml_path = runtime_dir + "\\ggml-base.dll";
    const std::string ggml_loader_path = runtime_dir + "\\ggml.dll";
    api.llama = LoadLibraryA(llama_path.c_str());
    api.ggml_base = LoadLibraryA(ggml_path.c_str());
    api.ggml_loader = LoadLibraryA(ggml_loader_path.c_str());
    if (!api.llama || !api.ggml_base || !api.ggml_loader) {
        std::cerr << "Could not load llama.cpp runtime DLLs from " << runtime_dir << "\n";
        return false;
    }
    return load_symbol(api.llama, "llama_backend_init", api.backend_init)
        && load_symbol(api.llama, "llama_backend_free", api.backend_free)
        && load_symbol(api.llama, "llama_model_default_params", api.model_default_params)
        && load_symbol(api.llama, "llama_context_default_params", api.context_default_params)
        && load_symbol(api.llama, "llama_model_load_from_file", api.model_load_from_file)
        && load_symbol(api.llama, "llama_model_free", api.model_free)
        && load_symbol(api.llama, "llama_model_get_vocab", api.model_get_vocab)
        && load_symbol(api.llama, "llama_model_n_layer", api.model_n_layer)
        && load_symbol(api.llama, "llama_model_n_embd", api.model_n_embd)
        && load_symbol(api.llama, "llama_init_from_model", api.init_from_model)
        && load_symbol(api.llama, "llama_free", api.context_free)
        && load_symbol(api.llama, "llama_tokenize", api.tokenize)
        && load_symbol(api.llama, "llama_batch_get_one", api.batch_get_one)
        && load_symbol(api.llama, "llama_decode", api.decode)
        && load_symbol(api.ggml_base, "ggml_backend_tensor_get", api.tensor_get)
        && load_symbol(api.ggml_loader, "ggml_backend_load_all_from_path", api.load_backends_from_path);
}

struct Capture {
    Api * api = nullptr;
    int32_t n_tokens = 0;
    int32_t n_embd = 0;
    std::vector<std::string> names;
    std::vector<std::vector<float>> states;
};

static std::string json_escape(const std::string & value) {
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (const char ch : value) {
        if (ch == '\\' || ch == '"') escaped.push_back('\\');
        escaped.push_back(ch);
    }
    return escaped;
}

static bool capture_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * capture = static_cast<Capture *>(user_data);
    const bool is_layer_output = std::strncmp(tensor->name, "l_out", 5) == 0;
    if (ask) {
        return is_layer_output;
    }
    if (!is_layer_output || tensor->type != GGML_TYPE_F32 || tensor->ne[0] != capture->n_embd || tensor->ne[1] < 1) {
        return true;
    }
    std::vector<float> state(static_cast<size_t>(capture->n_embd));
    // llama.cpp keeps the final block output only at the requested output token;
    // earlier blocks retain all prompt positions. In both cases the final row is
    // the same last-token state requested by this experiment.
    const size_t last_token_offset = static_cast<size_t>(tensor->nb[1]) * static_cast<size_t>(tensor->ne[1] - 1);
    capture->api->tensor_get(tensor, state.data(), last_token_offset, state.size() * sizeof(float));
    capture->names.emplace_back(tensor->name);
    capture->states.emplace_back(std::move(state));
    return true;
}

static bool write_capture(const std::string & output_prefix, const Capture & capture, int32_t n_layers, const std::string & model_path) {
    const std::string binary_path = output_prefix + ".f32";
    const std::string metadata_path = output_prefix + ".json";
    std::ofstream binary(binary_path, std::ios::binary);
    if (!binary) {
        std::cerr << "Could not write " << binary_path << "\n";
        return false;
    }
    for (const auto & state : capture.states) {
        binary.write(reinterpret_cast<const char *>(state.data()), static_cast<std::streamsize>(state.size() * sizeof(float)));
    }
    std::ofstream metadata(metadata_path);
    if (!metadata) {
        std::cerr << "Could not write " << metadata_path << "\n";
        return false;
    }
    metadata << "{\n"
             << "  \"format\": \"layer_capture_f32_row_major_v1\",\n"
             << "  \"model_path\": \"" << json_escape(model_path) << "\",\n"
             << "  \"n_model_layers\": " << n_layers << ",\n"
             << "  \"n_captured_layers\": " << capture.states.size() << ",\n"
             << "  \"n_embd\": " << capture.n_embd << ",\n"
             << "  \"n_tokens\": " << capture.n_tokens << ",\n"
             << "  \"tensor_names\": [";
    for (size_t i = 0; i < capture.names.size(); ++i) {
        if (i) metadata << ", ";
        metadata << "\"" << json_escape(capture.names[i]) << "\"";
    }
    metadata << "]\n}\n";
    return true;
}

static bool capture_prompt(
    Api & api,
    llama_model * model,
    const std::string & model_path,
    const std::string & prompt_path,
    const std::string & output_prefix
) {
    std::ifstream prompt_file(prompt_path, std::ios::binary);
    const std::string prompt((std::istreambuf_iterator<char>(prompt_file)), std::istreambuf_iterator<char>());
    if (prompt.empty()) {
        std::cerr << "Prompt is empty or unreadable: " << prompt_path << "\n";
        return false;
    }
    const llama_vocab * vocab = api.model_get_vocab(model);
    std::vector<llama_token> tokens(prompt.size() + 16);
    int32_t n_tokens = api.tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
    if (n_tokens < 0) {
        tokens.resize(static_cast<size_t>(-n_tokens));
        n_tokens = api.tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
    }
    if (n_tokens <= 0) {
        std::cerr << "Tokenization failed for " << prompt_path << "\n";
        return false;
    }
    tokens.resize(static_cast<size_t>(n_tokens));
    Capture capture;
    capture.api = &api;
    capture.n_tokens = n_tokens;
    capture.n_embd = api.model_n_embd(model);
    llama_context_params context_params = api.context_default_params();
    context_params.n_ctx = std::max<uint32_t>(2048, static_cast<uint32_t>(n_tokens + 32));
    context_params.n_batch = static_cast<uint32_t>(n_tokens);
    context_params.n_ubatch = static_cast<uint32_t>(n_tokens);
    context_params.cb_eval = capture_callback;
    context_params.cb_eval_user_data = &capture;
    context_params.no_perf = true;
    llama_context * context = api.init_from_model(model, context_params);
    if (!context) {
        std::cerr << "Failed to create context for " << prompt_path << "\n";
        return false;
    }
    const llama_batch batch = api.batch_get_one(tokens.data(), n_tokens);
    const int decode_status = api.decode(context, batch);
    const int32_t n_layers = api.model_n_layer(model);
    const bool ok = decode_status == 0 && static_cast<int32_t>(capture.states.size()) == n_layers
        && write_capture(output_prefix, capture, n_layers, model_path);
    std::cout << "prompt=" << prompt_path << " decode_status=" << decode_status << " n_tokens=" << n_tokens
              << " n_embd=" << capture.n_embd << " n_model_layers=" << n_layers
              << " n_captured_layers=" << capture.states.size() << "\n";
    api.context_free(context);
    return ok;
}

int main(int argc, char ** argv) {
    const bool batch_mode = argc == 6 && std::string(argv[3]) == "--batch";
    if (argc != 5 && !batch_mode) {
        std::cerr << "Usage: gemma3_layer_capture <runtime_dir> <model.gguf> <prompt.txt> <output_prefix>\n"
                  << "   or: gemma3_layer_capture <runtime_dir> <model.gguf> --batch <prompt_manifest.tsv> <output_dir>\n";
        return 2;
    }
    const std::string runtime_dir = argv[1];
    const std::string model_path = argv[2];
    std::vector<std::pair<std::string, std::string>> jobs;
    if (batch_mode) {
        const std::string manifest_path = argv[4];
        const std::string output_dir = argv[5];
        std::ifstream manifest(manifest_path);
        std::string line;
        while (std::getline(manifest, line)) {
            const size_t tab = line.find('\t');
            if (tab == std::string::npos || tab == 0 || tab == line.size() - 1) continue;
            jobs.emplace_back(line.substr(0, tab), line.substr(tab + 1));
        }
        if (jobs.empty()) {
            std::cerr << "Batch manifest is empty or invalid: " << manifest_path << "\n";
            return 2;
        }
        std::filesystem::create_directories(output_dir);
        for (auto & job : jobs) job.first = output_dir + "\\" + job.first;
    } else {
        jobs.emplace_back(argv[4], argv[3]);
    }

    Api api;
    if (!initialise_api(runtime_dir, api)) return 1;
    api.load_backends_from_path(runtime_dir.c_str());
    api.backend_init();
    llama_model_params model_params = api.model_default_params();
    model_params.n_gpu_layers = -1;
    llama_model * model = api.model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        std::cerr << "Failed to load model.\n";
        api.backend_free();
        return 1;
    }
    bool ok = true;
    for (const auto & job : jobs) {
        ok = capture_prompt(api, model, model_path, job.second, job.first) && ok;
    }
    if (!ok) {
        std::cerr << "One or more captures were incomplete; affected outputs are not qualified.\n";
    }
    api.model_free(model);
    api.backend_free();
    FreeLibrary(api.ggml_base);
    FreeLibrary(api.ggml_loader);
    FreeLibrary(api.llama);
    return ok ? 0 : 1;
}
