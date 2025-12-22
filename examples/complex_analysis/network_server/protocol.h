#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <stdint.h>
#include <stddef.h>

// Protocol constants
#define MAX_PACKET_SIZE 4096
#define MAX_PAYLOAD_SIZE 3072
#define HEADER_SIZE 24
#define MAX_USERNAME_LEN 64
#define MAX_PASSWORD_LEN 128

// Message types
typedef enum {
    MSG_TYPE_AUTH = 0x01,
    MSG_TYPE_DATA = 0x02,
    MSG_TYPE_CONTROL = 0x03,
    MSG_TYPE_HEARTBEAT = 0x04,
    MSG_TYPE_ERROR = 0xFF
} MessageType;

// Packet header structure
typedef struct {
    uint32_t magic;           // Magic number: 0xDEADBEEF
    uint16_t version;         // Protocol version
    uint8_t  msg_type;        // Message type
    uint8_t  flags;           // Flags
    uint32_t sequence;        // Sequence number
    uint32_t payload_length;  // Payload length
    uint32_t checksum;        // CRC32 checksum
    uint32_t reserved;        // Reserved for future use
} __attribute__((packed)) PacketHeader;

// Authentication request
typedef struct {
    char username[MAX_USERNAME_LEN];
    char password[MAX_PASSWORD_LEN];
    uint32_t timestamp;
    uint32_t client_id;
} __attribute__((packed)) AuthRequest;

// Data packet payload
typedef struct {
    uint32_t data_id;
    uint32_t data_type;
    uint32_t data_length;
    uint8_t  data[MAX_PAYLOAD_SIZE];
} __attribute__((packed)) DataPayload;

// Control command
typedef struct {
    uint32_t command_id;
    uint32_t param_count;
    uint32_t params[16];
} __attribute__((packed)) ControlCommand;

// Function declarations
int validate_packet_header(const PacketHeader *header);
int parse_auth_request(const uint8_t *data, size_t len, AuthRequest *auth);
int parse_data_payload(const uint8_t *data, size_t len, DataPayload *payload);
int parse_control_command(const uint8_t *data, size_t len, ControlCommand *cmd);
uint32_t calculate_checksum(const uint8_t *data, size_t len);

#endif // PROTOCOL_H

