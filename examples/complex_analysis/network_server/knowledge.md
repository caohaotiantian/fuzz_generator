# Network Protocol Knowledge Base

## Protocol Overview

This is a custom binary network protocol with the following characteristics:

### Packet Structure
- **Header**: 24 bytes (PacketHeader struct)
- **Payload**: Variable length, max 3072 bytes
- **Total max size**: 4096 bytes

### Header Fields
- `magic`: 0xDEADBEEF (4 bytes) - Protocol identifier
- `version`: Protocol version number (2 bytes)
- `msg_type`: Message type (1 byte)
- `flags`: Protocol flags (1 byte)
- `sequence`: Packet sequence number (4 bytes)
- `payload_length`: Length of payload data (4 bytes)
- `checksum`: CRC32 checksum of payload (4 bytes)
- `reserved`: Reserved for future use (4 bytes)

### Message Types
1. **MSG_TYPE_AUTH (0x01)**: Authentication request
   - Contains username (max 64 bytes)
   - Contains password (max 128 bytes)
   - Contains timestamp and client_id

2. **MSG_TYPE_DATA (0x02)**: Data transfer
   - Requires prior authentication
   - Contains data_id, data_type, data_length
   - Actual data follows (max 3072 bytes)

3. **MSG_TYPE_CONTROL (0x03)**: Control commands
   - Requires prior authentication
   - Contains command_id and parameter count
   - Up to 16 uint32_t parameters

4. **MSG_TYPE_HEARTBEAT (0x04)**: Keep-alive
   - No authentication required
   - Empty payload

## Security Considerations

### Input Validation
1. **Packet size**: Must be at least 24 bytes (header size)
2. **Magic number**: Must be 0xDEADBEEF
3. **Version**: Must be <= 2
4. **Message type**: Must be valid (0x01-0x04, 0xFF)
5. **Payload length**: Must be <= 3072 bytes
6. **Checksum**: Must match calculated CRC32

### Authentication
- Authentication required before DATA or CONTROL messages
- Username must be non-empty
- Password must be at least 8 characters
- Timestamp must be within ±5 minutes of server time

### Sequence Numbers
- Must increment by 1 for each packet
- Gaps allowed (packet loss tolerance)
- Backwards sequence numbers rejected

## Data Flow Analysis

### process_packet Function
**Entry point for all packets**

Input: `packet_data` (uint8_t*), `packet_len` (size_t)

Data flow:
1. packet_data → PacketHeader (cast)
2. PacketHeader → validate_packet_header()
3. PacketHeader.sequence → verify_sequence()
4. packet_data + sizeof(PacketHeader) → payload
5. payload + header.payload_length → calculate_checksum()
6. Based on msg_type:
   - MSG_TYPE_AUTH → handle_auth_message() → parse_auth_request()
   - MSG_TYPE_DATA → handle_data_message() → parse_data_payload()
   - MSG_TYPE_CONTROL → handle_control_message() → parse_control_command()

### Critical Parameters

For **process_packet**:
- `packet_data`: Raw packet bytes (untrusted input)
- `packet_len`: Length of packet data
- Constraints:
  - packet_len >= 24 (sizeof(PacketHeader))
  - packet_len >= 24 + header.payload_length
  - header.magic == 0xDEADBEEF
  - header.payload_length <= 3072

For **parse_auth_request**:
- `data`: Authentication payload
- `len`: Payload length
- Constraints:
  - len >= sizeof(AuthRequest) (200 bytes)
  - username must be null-terminated
  - password must be >= 8 chars
  - timestamp within ±300 seconds

For **parse_data_payload**:
- `data`: Data payload
- `len`: Payload length
- Constraints:
  - len >= 12 (header fields)
  - data_length <= 3072
  - len >= 12 + data_length

For **parse_control_command**:
- `data`: Control command payload
- `len`: Payload length
- Constraints:
  - len >= 8 (command_id + param_count)
  - param_count <= 16
  - len >= 8 + param_count * 4

## Fuzzing Targets

### High Priority
1. **process_packet**: Main entry point, complex validation logic
2. **parse_auth_request**: String handling, timestamp validation
3. **parse_data_payload**: Variable-length data, memcpy operations

### Medium Priority
4. **parse_control_command**: Array handling, parameter validation
5. **validate_packet_header**: Header field validation

### Attack Vectors
- Integer overflow in length calculations
- Buffer overflow in memcpy operations
- Off-by-one errors in array indexing
- Timestamp manipulation
- Sequence number manipulation
- Checksum bypass attempts

