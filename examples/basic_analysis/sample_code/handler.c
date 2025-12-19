/**
 * Sample C code for fuzz_generator demonstration.
 * 
 * This file contains a simple request handler function that
 * can be analyzed to generate a fuzz test data model.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/**
 * Process an incoming request buffer.
 * 
 * @param buffer  Input buffer containing request data
 * @param length  Length of valid data in buffer
 * @return 0 on success, -1 on error
 * 
 * This function is a typical entry point for fuzz testing:
 * - Takes external input (buffer)
 * - Has length constraints (0 < length < 256)
 * - Performs string operations
 */
int process_request(char* buffer, int length) {
    char local_buf[256];
    
    // Validate input parameters
    if (buffer == NULL) {
        fprintf(stderr, "Error: NULL buffer\n");
        return -1;
    }
    
    // Length validation
    if (length <= 0 || length >= 256) {
        fprintf(stderr, "Error: Invalid length %d\n", length);
        return -1;
    }
    
    // Copy to local buffer (potential vulnerability if length check is bypassed)
    strncpy(local_buf, buffer, length);
    local_buf[length] = '\0';
    
    // Process the request
    printf("Processing request: %s\n", local_buf);
    
    return 0;
}

/**
 * Parse a key-value header line.
 * 
 * @param line    Input line in "Key: Value" format
 * @param key     Output buffer for key (caller must provide)
 * @param value   Output buffer for value (caller must provide)
 * @return 0 on success, -1 on error
 */
int parse_header_line(const char* line, char* key, char* value) {
    if (line == NULL || key == NULL || value == NULL) {
        return -1;
    }
    
    // Find the colon separator
    const char* colon = strchr(line, ':');
    if (colon == NULL) {
        return -1;
    }
    
    // Extract key
    size_t key_len = colon - line;
    strncpy(key, line, key_len);
    key[key_len] = '\0';
    
    // Skip colon and whitespace
    const char* val_start = colon + 1;
    while (*val_start == ' ' || *val_start == '\t') {
        val_start++;
    }
    
    // Extract value
    strcpy(value, val_start);
    
    return 0;
}

/**
 * Main function for standalone testing.
 */
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <message>\n", argv[0]);
        return 1;
    }
    
    char* message = argv[1];
    int result = process_request(message, strlen(message));
    
    printf("Result: %d\n", result);
    return result;
}

