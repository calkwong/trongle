#pragma once

#include <string>
#include <unordered_map>
#include <vector>

using Entity = std::unordered_map<std::string, std::string>;

const std::vector<std::string> lpv_keys = {
    "box_maxs",
    "box_mins",
    "origin",
    "handshake",
    "light_probe_size_x",
    "light_probe_size_y",
    "light_probe_size_z",
    "array_index",
    "light_probe_atlas_x",
    "light_probe_atlas_y",
    "light_probe_atlas_z",
};

std::vector<Entity> parse_and_extract_lpvs(std::string entity_file);
