#include "protocol.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Session state
typedef struct {
    int authenticated;
    uint32_t client_id;
    char username[MAX_USERNAME_LEN];
    uint32_t last_sequence;
    time_t last_activity;
} SessionState;

static SessionState current_session = {0};

// Forward declarations
static int handle_auth_message(const uint8_t *payload, size_t len);
static int handle_data_message(const uint8_t *payload, size_t len);
static int handle_control_message(const uint8_t *payload, size_t len);
static int verify_sequence(uint32_t sequence);

/**
 * Main packet processing function
 * This is the entry point for all incoming packets
 */
int process_packet(const uint8_t *packet_data, size_t packet_len) {
    if (packet_data == NULL) {
        fprintf(stderr, "Error: NULL packet data\n");
        return -1;
    }
    
    // Check minimum packet size
    if (packet_len < sizeof(PacketHeader)) {
        fprintf(stderr, "Error: Packet too small: %zu bytes\n", packet_len);
        return -1;
    }
    
    // Parse packet header
    const PacketHeader *header = (const PacketHeader *)packet_data;
    
    // Validate header
    if (validate_packet_header(header) != 0) {
        fprintf(stderr, "Error: Invalid packet header\n");
        return -1;
    }
    
    // Verify sequence number
    if (verify_sequence(header->sequence) != 0) {
        fprintf(stderr, "Error: Invalid sequence number: %u\n", header->sequence);
        return -1;
    }
    
    // Check total packet size
    size_t expected_size = sizeof(PacketHeader) + header->payload_length;
    if (packet_len < expected_size) {
        fprintf(stderr, "Error: Incomplete packet: expected %zu, got %zu\n",
                expected_size, packet_len);
        return -1;
    }
    
    // Verify checksum
    const uint8_t *payload = packet_data + sizeof(PacketHeader);
    uint32_t calculated_checksum = calculate_checksum(payload, header->payload_length);
    if (calculated_checksum != header->checksum) {
        fprintf(stderr, "Error: Checksum mismatch: expected 0x%08X, got 0x%08X\n",
                header->checksum, calculated_checksum);
        return -1;
    }
    
    // Update session activity
    current_session.last_activity = time(NULL);
    current_session.last_sequence = header->sequence;
    
    // Dispatch based on message type
    int result = 0;
    switch (header->msg_type) {
        case MSG_TYPE_AUTH:
            result = handle_auth_message(payload, header->payload_length);
            break;
            
        case MSG_TYPE_DATA:
            if (!current_session.authenticated) {
                fprintf(stderr, "Error: Data message received before authentication\n");
                return -1;
            }
            result = handle_data_message(payload, header->payload_length);
            break;
            
        case MSG_TYPE_CONTROL:
            if (!current_session.authenticated) {
                fprintf(stderr, "Error: Control message received before authentication\n");
                return -1;
            }
            result = handle_control_message(payload, header->payload_length);
            break;
            
        case MSG_TYPE_HEARTBEAT:
            // Heartbeat is always allowed
            printf("Heartbeat received from client %u\n", current_session.client_id);
            result = 0;
            break;
            
        default:
            fprintf(stderr, "Error: Unknown message type: 0x%02X\n", header->msg_type);
            result = -1;
            break;
    }
    
    return result;
}

static int handle_auth_message(const uint8_t *payload, size_t len) {
    AuthRequest auth;
    
    if (parse_auth_request(payload, len, &auth) != 0) {
        fprintf(stderr, "Error: Failed to parse auth request\n");
        return -1;
    }
    
    // TODO: Implement actual authentication logic
    // For now, accept any valid auth request
    printf("Authentication request from user: %s (client_id: %u)\n",
           auth.username, auth.client_id);
    
    // Update session state
    current_session.authenticated = 1;
    current_session.client_id = auth.client_id;
    strncpy(current_session.username, auth.username, MAX_USERNAME_LEN - 1);
    current_session.username[MAX_USERNAME_LEN - 1] = '\0';
    
    return 0;
}

static int handle_data_message(const uint8_t *payload, size_t len) {
    DataPayload data;
    
    if (parse_data_payload(payload, len, &data) != 0) {
        fprintf(stderr, "Error: Failed to parse data payload\n");
        return -1;
    }
    
    printf("Data message received: id=%u, type=%u, length=%u\n",
           data.data_id, data.data_type, data.data_length);
    
    // TODO: Process data based on data_type
    // For now, just log it
    
    return 0;
}

static int handle_control_message(const uint8_t *payload, size_t len) {
    ControlCommand cmd;
    
    if (parse_control_command(payload, len, &cmd) != 0) {
        fprintf(stderr, "Error: Failed to parse control command\n");
        return -1;
    }
    
    printf("Control command received: id=%u, param_count=%u\n",
           cmd.command_id, cmd.param_count);
    
    // TODO: Execute control command
    // For now, just log parameters
    for (uint32_t i = 0; i < cmd.param_count; i++) {
        printf("  param[%u] = 0x%08X\n", i, cmd.params[i]);
    }
    
    return 0;
}

static int verify_sequence(uint32_t sequence) {
    // First packet can have any sequence number
    if (current_session.last_sequence == 0) {
        return 0;
    }
    
    // Sequence number should increment by 1
    uint32_t expected = current_session.last_sequence + 1;
    if (sequence != expected) {
        fprintf(stderr, "Warning: Sequence number gap: expected %u, got %u\n",
                expected, sequence);
        // Allow sequence number gaps (packet loss)
        // But reject backwards sequence numbers
        if (sequence < current_session.last_sequence) {
            return -1;
        }
    }
    
    return 0;
}

// Helper function to reset session
void reset_session(void) {
    memset(&current_session, 0, sizeof(SessionState));
}

// Helper function to get session info
int get_session_info(char *buffer, size_t buffer_size) {
    if (buffer == NULL || buffer_size == 0) {
        return -1;
    }
    
    snprintf(buffer, buffer_size,
             "Session: authenticated=%d, client_id=%u, username=%s, last_seq=%u",
             current_session.authenticated,
             current_session.client_id,
             current_session.username,
             current_session.last_sequence);
    
    return 0;
}

