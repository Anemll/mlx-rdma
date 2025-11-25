// Simple utility to discover InfiniBand devices on macOS
// Compile with: c++ -o discover_ibv_devices discover_ibv_devices.cpp -lrdma
// Run with: ./discover_ibv_devices

#include <infiniband/verbs.h>
#include <iostream>
#include <vector>
#include <string>

// Helper function to decode MTU value to bytes
int mtu_to_bytes(int mtu) {
    switch (mtu) {
        case 1: return 256;
        case 2: return 512;
        case 3: return 1024;
        case 4: return 2048;
        case 5: return 4096;
        default: return -1;
    }
}

// Helper function to get port state name
const char* port_state_name(ibv_port_state state) {
    switch (state) {
        case IBV_PORT_DOWN: return "DOWN";
        case IBV_PORT_INIT: return "INIT";
        case IBV_PORT_ARMED: return "ARMED";
        case IBV_PORT_ACTIVE: return "ACTIVE";
        default: return "UNKNOWN";
    }
}

int main() {
    int num_devices = 0;
    ibv_device** devices = ibv_get_device_list(&num_devices);
    
    if (!devices) {
        std::cerr << "Error: Could not get device list" << std::endl;
        std::cerr << "Make sure libibverbs is installed and InfiniBand hardware is available" << std::endl;
        return 1;
    }
    
    if (num_devices == 0) {
        std::cout << "No InfiniBand devices found." << std::endl;
        std::cout << "\nThis could mean:" << std::endl;
        std::cout << "  1. No InfiniBand hardware is installed" << std::endl;
        std::cout << "  2. InfiniBand drivers are not loaded" << std::endl;
        std::cout << "  3. macOS version doesn't support InfiniBand (requires macOS 26.2+)" << std::endl;
        ibv_free_device_list(devices);
        return 1;
    }
    
    std::cout << "Found " << num_devices << " InfiniBand device(s):\n" << std::endl;
    
    std::vector<std::string> device_names;
    for (int i = 0; i < num_devices; i++) {
        const char* name = ibv_get_device_name(devices[i]);
        device_names.push_back(name);
        
        std::cout << "Device " << i << ": " << name << std::endl;
        
        // Try to get more information
        ibv_context* ctx = ibv_open_device(devices[i]);
        if (ctx) {
            ibv_device_attr attr;
            if (ibv_query_device(ctx, &attr) == 0) {
                std::cout << "  - Vendor ID: 0x" << std::hex << attr.vendor_id << std::dec << std::endl;
                std::cout << "  - Device ID: 0x" << std::hex << attr.vendor_part_id << std::dec << std::endl;
                std::cout << "  - Max MR size: " << attr.max_mr_size << " bytes" << std::endl;
                std::cout << "  - Max QP: " << attr.max_qp << std::endl;
                std::cout << "  - Max CQ: " << attr.max_cq << std::endl;
            }
            
            // Query port information
            ibv_port_attr port_attr;
            if (ibv_query_port(ctx, 1, &port_attr) == 0) {
                std::cout << "  - Port 1 LID: " << port_attr.lid << std::endl;
                int mtu_bytes = mtu_to_bytes(port_attr.max_mtu);
                std::cout << "  - Port 1 State: " << (int)port_attr.state 
                          << " (" << port_state_name(port_attr.state) << ")" << std::endl;
                std::cout << "  - Port 1 MTU: " << (int)port_attr.max_mtu;
                if (mtu_bytes > 0) {
                    std::cout << " (" << mtu_bytes << " bytes)";
                }
                std::cout << std::endl;
            }
            
            ibv_close_device(ctx);
        }
        std::cout << std::endl;
    }
    
    // Generate example JSON configuration
    if (device_names.size() > 0) {
        std::cout << "\n=== Example JSON Configuration ===" << std::endl;
        std::cout << "For 2-node setup:\n" << std::endl;
        std::cout << "{\n";
        std::cout << "  \"0\": [null, \"" << device_names[0] << "\"],\n";
        std::cout << "  \"1\": [\"" << device_names[0] << "\", null]\n";
        std::cout << "}\n" << std::endl;
        
        if (device_names.size() >= 2) {
            std::cout << "For 4-node setup:\n" << std::endl;
            std::cout << "{\n";
            std::cout << "  \"0\": [null, \"" << device_names[0] << "\", \"" << device_names[1] << "\", \"" << device_names[0] << "\"],\n";
            std::cout << "  \"1\": [\"" << device_names[0] << "\", null, \"" << device_names[1] << "\", \"" << device_names[0] << "\"],\n";
            std::cout << "  \"2\": [\"" << device_names[0] << "\", \"" << device_names[1] << "\", null, \"" << device_names[0] << "\"],\n";
            std::cout << "  \"3\": [\"" << device_names[0] << "\", \"" << device_names[1] << "\", \"" << device_names[0] << "\", null]\n";
            std::cout << "}\n" << std::endl;
        }
    }
    
    ibv_free_device_list(devices);
    return 0;
}

