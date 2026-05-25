#include "mlir/Tools/Plugins/PassPlugin.h"
#include <string>
#include <iostream>

extern "C" bool loadMlirPassPlugin(const char* pluginPath) {
    if (!pluginPath) return false;
    
    std::string path(pluginPath);
    auto plugin = mlir::PassPlugin::load(path);
    if (!plugin) {
        std::cerr << "[JIT] Failed to load MLIR Pass Plugin: " << path << std::endl;
        return false;
    }
    
    plugin.get().registerPassRegistryCallbacks();
    return true;
}
