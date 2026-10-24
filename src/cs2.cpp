#include "cs2.h"

#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <filesystem>

namespace
{
const char* WS = " \t\r\n";

std::string trim(const std::string& s)
{
    size_t b = s.find_first_not_of(WS);
    if (b == std::string::npos)
        return "";
    size_t e = s.find_last_not_of(WS);
    return s.substr(b, e - b + 1);
}

bool is_header(const std::string& line)
{
    return line.size() >= 8 && line.compare(0, 4, "====") == 0 && line.compare(line.size() - 4, 4, "====") == 0;
}

std::vector<Entity> parse(std::istream& in)
{
    std::vector<Entity> entities;
    std::string line;

    while (std::getline(in, line))
    {
        std::string t = trim(line);
        if (t.empty())
            continue;

        if (is_header(t))
        {
            entities.emplace_back();
            continue;
        }

        // Defensive
        if (entities.empty())
            entities.emplace_back();

        // Key is the first string, value is everything after it
        size_t keyEnd = t.find_first_of(" \t");
        std::string key = t.substr(0, keyEnd);
        std::string value =
            (keyEnd == std::string::npos) ? "" : trim(t.substr(keyEnd));

        // Multi-line value: opened by """ and closed by a line of exactly """
        if (value == "\"\"\"")
        {
            std::string block = value;
            while (std::getline(in, line))
            {
                std::string text = trim(line);
                block += "\n" + line;
                if (text == "\"\"\"")
                    break;
            }
            value = block;
        }

        entities.back()[key] = value;
    }
    return entities;
}

bool is_light_probe_volume(const Entity& e)
{
    auto it = e.find("classname");
    return (it != e.end()) && (it->second.find("light_probe_volume") != std::string::npos);
}
} // namespace

std::vector<Entity> parse_and_extract_lpvs(std::string entity_file)
{
    std::filesystem::path entity_path = "../assets/" + entity_file;

    std::ifstream f(entity_path);
    std::vector<Entity> probes{};
    std::vector<Entity> entities = parse(f);

    // Light probe volumes reduced to only the lpv_keys fields
    for (const Entity& e : entities)
    {
        if (!is_light_probe_volume(e))
            continue;
        Entity p;
        for (const std::string& k : lpv_keys)
        {
            auto it = e.find(k);
            if (it != e.end())
                p[k] = it->second;
        }
        probes.push_back(std::move(p));
    }

    return probes;
}
