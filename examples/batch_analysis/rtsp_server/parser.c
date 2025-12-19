/**
 * RTSP Protocol Parser
 * 
 * Parsing functions for RTSP protocol messages.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#define MAX_LINE_LENGTH 1024
#define MAX_HEADERS 32

/**
 * Header structure
 */
typedef struct {
    char name[64];
    char value[256];
} rtsp_header_t;

/**
 * Parsed RTSP message
 */
typedef struct {
    char method[16];
    char uri[512];
    char version[16];
    rtsp_header_t headers[MAX_HEADERS];
    int header_count;
    char* body;
    int body_length;
} rtsp_message_t;

/**
 * Parse an RTSP request line.
 * 
 * @param line    Input line (e.g., "OPTIONS rtsp://host/path RTSP/1.0")
 * @param method  Output: method string
 * @param uri     Output: URI string
 * @param version Output: version string
 * @return 0 on success, -1 on error
 */
int parse_rtsp_request(const char* line, char* method, char* uri, char* version) {
    if (line == NULL || method == NULL || uri == NULL || version == NULL) {
        return -1;
    }
    
    // Skip leading whitespace
    while (*line && isspace((unsigned char)*line)) {
        line++;
    }
    
    // Parse method
    const char* p = line;
    while (*p && !isspace((unsigned char)*p)) {
        p++;
    }
    
    size_t method_len = p - line;
    if (method_len == 0 || method_len >= 16) {
        return -1;
    }
    
    strncpy(method, line, method_len);
    method[method_len] = '\0';
    
    // Skip whitespace
    while (*p && isspace((unsigned char)*p)) {
        p++;
    }
    
    // Parse URI
    line = p;
    while (*p && !isspace((unsigned char)*p)) {
        p++;
    }
    
    size_t uri_len = p - line;
    if (uri_len == 0 || uri_len >= 512) {
        return -1;
    }
    
    strncpy(uri, line, uri_len);
    uri[uri_len] = '\0';
    
    // Skip whitespace
    while (*p && isspace((unsigned char)*p)) {
        p++;
    }
    
    // Parse version
    line = p;
    while (*p && !isspace((unsigned char)*p) && *p != '\r' && *p != '\n') {
        p++;
    }
    
    size_t version_len = p - line;
    if (version_len == 0 || version_len >= 16) {
        return -1;
    }
    
    strncpy(version, line, version_len);
    version[version_len] = '\0';
    
    return 0;
}

/**
 * Parse a single RTSP header line.
 * 
 * @param line   Input line (e.g., "CSeq: 1")
 * @param name   Output: header name
 * @param value  Output: header value
 * @return 0 on success, -1 on error
 */
int parse_header_line(const char* line, char* name, char* value) {
    if (line == NULL || name == NULL || value == NULL) {
        return -1;
    }
    
    // Find colon
    const char* colon = strchr(line, ':');
    if (colon == NULL) {
        return -1;
    }
    
    // Copy name
    size_t name_len = colon - line;
    if (name_len == 0 || name_len >= 64) {
        return -1;
    }
    
    strncpy(name, line, name_len);
    name[name_len] = '\0';
    
    // Skip colon and whitespace
    const char* p = colon + 1;
    while (*p && (*p == ' ' || *p == '\t')) {
        p++;
    }
    
    // Copy value (remove trailing whitespace)
    size_t value_len = strlen(p);
    while (value_len > 0 && (p[value_len - 1] == '\r' || 
                             p[value_len - 1] == '\n' ||
                             p[value_len - 1] == ' ')) {
        value_len--;
    }
    
    if (value_len >= 256) {
        value_len = 255;
    }
    
    strncpy(value, p, value_len);
    value[value_len] = '\0';
    
    return 0;
}

/**
 * Extract CSeq from headers.
 * 
 * @param message  Parsed RTSP message
 * @return CSeq value, or -1 if not found
 */
int extract_cseq(const rtsp_message_t* message) {
    if (message == NULL) {
        return -1;
    }
    
    for (int i = 0; i < message->header_count; i++) {
        if (strcasecmp(message->headers[i].name, "CSeq") == 0) {
            return atoi(message->headers[i].value);
        }
    }
    
    return -1;
}

