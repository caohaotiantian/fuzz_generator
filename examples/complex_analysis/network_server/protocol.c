#include "protocol.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// CRC32 lookup table
static uint32_t crc32_table[256];
static int crc32_table_initialized = 0;

static void init_crc32_table(void) {
    uint32_t polynomial = 0xEDB88320;
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (uint32_t j = 0; j < 8; j++) {
            if (crc & 1) {
                crc = (crc >> 1) ^ polynomial;
            } else {
                crc >>= 1;
            }
        }
        crc32_table[i] = crc;
    }
    crc32_table_initialized = 1;
}

uint32_t calculate_checksum(const uint8_t *data, size_t len) {
    if (!crc32_table_initialized) {
        init_crc32_table();
    }
    
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) {
        uint8_t index = (crc ^ data[i]) & 0xFF;
        crc = (crc >> 8) ^ crc32_table[index];
    }
    return ~crc;
}

int validate_packet_header(const PacketHeader *header) {
    if (header == NULL) {
        fprintf(stderr, "Error: NULL header pointer\n");
        return -1;
    }
    
    // Check magic number
    if (header->magic != 0xDEADBEEF) {
        fprintf(stderr, "Error: Invalid magic number: 0x%08X\n", header->magic);
        return -1;
    }
    
    // Check version
    if (header->version > 2) {
        fprintf(stderr, "Error: Unsupported version: %u\n", header->version);
        return -1;
    }
    
    // Check message type
    if (header->msg_type == 0 || header->msg_type > MSG_TYPE_ERROR) {
        fprintf(stderr, "Error: Invalid message type: 0x%02X\n", header->msg_type);
        return -1;
    }
    
    // Check payload length
    if (header->payload_length > MAX_PAYLOAD_SIZE) {
        fprintf(stderr, "Error: Payload too large: %u bytes\n", header->payload_length);
        return -1;
    }
    
    return 0;
}

int parse_auth_request(const uint8_t *data, size_t len, AuthRequest *auth) {
    if (data == NULL || auth == NULL) {
        fprintf(stderr, "Error: NULL pointer in parse_auth_request\n");
        return -1;
    }
    
    if (len < sizeof(AuthRequest)) {
        fprintf(stderr, "Error: Insufficient data for auth request: %zu bytes\n", len);
        return -1;
    }
    
    // Copy data to auth structure
    memcpy(auth, data, sizeof(AuthRequest));
    
    // Validate username (must be null-terminated)
    auth->username[MAX_USERNAME_LEN - 1] = '\0';
    if (strlen(auth->username) == 0) {
        fprintf(stderr, "Error: Empty username\n");
        return -1;
    }
    
    // Validate password (must be null-terminated)
    auth->password[MAX_PASSWORD_LEN - 1] = '\0';
    if (strlen(auth->password) < 8) {
        fprintf(stderr, "Error: Password too short\n");
        return -1;
    }
    
    // Validate timestamp (should be within reasonable range)
    uint32_t current_time = (uint32_t)time(NULL);
    if (auth->timestamp > current_time + 300 || auth->timestamp < current_time - 300) {
        fprintf(stderr, "Error: Invalid timestamp: %u\n", auth->timestamp);
        return -1;
    }
    
    return 0;
}

int parse_data_payload(const uint8_t *data, size_t len, DataPayload *payload) {
    if (data == NULL || payload == NULL) {
        fprintf(stderr, "Error: NULL pointer in parse_data_payload\n");
        return -1;
    }
    
    // Need at least the header fields
    if (len < 12) {
        fprintf(stderr, "Error: Data too short: %zu bytes\n", len);
        return -1;
    }
    
    // Parse header fields
    memcpy(&payload->data_id, data, 4);
    memcpy(&payload->data_type, data + 4, 4);
    memcpy(&payload->data_length, data + 8, 4);
    
    // Validate data length
    if (payload->data_length > MAX_PAYLOAD_SIZE) {
        fprintf(stderr, "Error: Data length too large: %u\n", payload->data_length);
        return -1;
    }
    
    if (len < 12 + payload->data_length) {
        fprintf(stderr, "Error: Insufficient data: expected %u, got %zu\n",
                12 + payload->data_length, len);
        return -1;
    }
    
    // Copy payload data
    memcpy(payload->data, data + 12, payload->data_length);
    
    return 0;
}

int parse_control_command(const uint8_t *data, size_t len, ControlCommand *cmd) {
    if (data == NULL || cmd == NULL) {
        fprintf(stderr, "Error: NULL pointer in parse_control_command\n");
        return -1;
    }
    
    if (len < 8) {
        fprintf(stderr, "Error: Control command too short: %zu bytes\n", len);
        return -1;
    }
    
    // Parse command header
    memcpy(&cmd->command_id, data, 4);
    memcpy(&cmd->param_count, data + 4, 4);
    
    // Validate parameter count
    if (cmd->param_count > 16) {
        fprintf(stderr, "Error: Too many parameters: %u\n", cmd->param_count);
        return -1;
    }
    
    if (len < 8 + cmd->param_count * 4) {
        fprintf(stderr, "Error: Insufficient data for parameters\n");
        return -1;
    }
    
    // Copy parameters
    for (uint32_t i = 0; i < cmd->param_count; i++) {
        memcpy(&cmd->params[i], data + 8 + i * 4, 4);
    }
    
    return 0;
}

