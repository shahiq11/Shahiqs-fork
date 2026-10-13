#include "graphics/host_gpu/debugFlags.h"

#include <cstdio>
#include <fstream>
#include <mutex>
#include <set>
#include <string>
#include <unordered_set>

namespace Libs::Graphics::DebugFlags {

namespace {

struct Flags {
	std::set<std::string, std::less<>> names;
	std::unordered_set<uint64_t>       skipped_shaders;
};

std::string Trim(std::string text) {
	const auto first = text.find_first_not_of(" \t\r\n");
	if (first == std::string::npos) return {};
	const auto last = text.find_last_not_of(" \t\r\n");
	return text.substr(first, last - first + 1u);
}

const Flags& Get() {
	static const Flags flags = [] {
		Flags result;
		std::ifstream file("kyty_debug_flags.txt");
		std::string   line;
		while (std::getline(file, line)) {
			if (const auto comment = line.find('#'); comment != std::string::npos) {
				line.resize(comment);
			}
			line = Trim(line);
			if (line.empty()) continue;
			const auto equals = line.find('=');
			const auto name   = Trim(line.substr(0, equals));
			const auto value  = equals == std::string::npos ? std::string {} : Trim(line.substr(equals + 1u));
			std::printf("Debug flag: %s%s%s\n", name.c_str(), value.empty() ? "" : "=", value.c_str());
			if (name == "skip_shader" && !value.empty()) {
				result.skipped_shaders.insert(std::stoull(value, nullptr, 16));
			} else {
				result.names.insert(name);
			}
		}
		std::fflush(stdout);
		return result;
	}();
	return flags;
}

} // namespace

bool Has(std::string_view name) {
	return Get().names.contains(name);
}

bool SkipShader(uint64_t shader_hash) {
	const auto& flags = Get();
	return !flags.skipped_shaders.empty() && flags.skipped_shaders.contains(shader_hash);
}

} // namespace Libs::Graphics::DebugFlags
