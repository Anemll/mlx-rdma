// Simpler C version to check AppleThunderboltRDMAPeerInterface count
// Compile with: cc -o check_rdma_peer_interface_simple check_rdma_peer_interface_simple.c -framework IOKit -framework CoreFoundation

#include <IOKit/IOKitLib.h>
#include <CoreFoundation/CoreFoundation.h>
#include <stdio.h>

int main() {
    // Get the root entry of the IOKit registry
    // IOKitDiagnostics is stored at the root, not in a specific service
    io_registry_entry_t rootEntry = IORegistryGetRootEntry(kIOMainPortDefault);
    
    if (rootEntry == 0) {
        fprintf(stderr, "Error: Could not get IOKit root entry\n");
        return 1;
    }
    
    // Get IOKitDiagnostics from root
    CFTypeRef diagnostics = IORegistryEntryCreateCFProperty(
        rootEntry,
        CFSTR("IOKitDiagnostics"),
        kCFAllocatorDefault,
        0
    );
    IOObjectRelease(rootEntry);
    
    if (!diagnostics) {
        fprintf(stderr, "Error: Could not get IOKitDiagnostics\n");
        return 1;
    }
    
    // Navigate: IOKitDiagnostics -> Classes -> AppleThunderboltRDMAPeerInterface
    CFDictionaryRef classes = CFDictionaryGetValue(
        (CFDictionaryRef)diagnostics,
        CFSTR("Classes")
    );
    
    if (classes) {
        CFNumberRef countRef = (CFNumberRef)CFDictionaryGetValue(
            classes,
            CFSTR("AppleThunderboltRDMAPeerInterface")
        );
        
        if (countRef) {
            int count = 0;
            CFNumberGetValue(countRef, kCFNumberIntType, &count);
            printf("AppleThunderboltRDMAPeerInterface count: %d\n", count);
        } else {
            printf("AppleThunderboltRDMAPeerInterface count: 0 (not found in registry)\n");
        }
    }
    
    CFRelease(diagnostics);
    return 0;
}

