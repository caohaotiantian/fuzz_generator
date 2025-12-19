/**
 * RTSP Session Manager
 * 
 * Session management for RTSP connections.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define MAX_SESSIONS 64
#define SESSION_ID_LEN 32

/**
 * Session state
 */
typedef enum {
    SESSION_STATE_INIT = 0,
    SESSION_STATE_READY,
    SESSION_STATE_PLAYING,
    SESSION_STATE_RECORDING,
    SESSION_STATE_TEARDOWN
} session_state_t;

/**
 * Session structure
 */
typedef struct {
    char session_id[SESSION_ID_LEN + 1];
    session_state_t state;
    char uri[512];
    time_t created_at;
    time_t last_activity;
    int transport_port;
} rtsp_session_t;

/**
 * Session manager
 */
static rtsp_session_t sessions[MAX_SESSIONS];
static int session_count = 0;

/**
 * Generate a new session ID.
 * 
 * @param buffer  Output buffer (at least SESSION_ID_LEN + 1 bytes)
 * @return Pointer to buffer
 */
char* generate_session_id(char* buffer) {
    if (buffer == NULL) {
        return NULL;
    }
    
    static const char charset[] = "0123456789ABCDEF";
    
    srand(time(NULL));
    
    for (int i = 0; i < SESSION_ID_LEN; i++) {
        buffer[i] = charset[rand() % (sizeof(charset) - 1)];
    }
    buffer[SESSION_ID_LEN] = '\0';
    
    return buffer;
}

/**
 * Create a new RTSP session.
 * 
 * @param uri             Stream URI
 * @param transport_port  Transport port number
 * @param session_id_out  Output: generated session ID
 * @return 0 on success, -1 on error
 */
int create_session(const char* uri, int transport_port, char* session_id_out) {
    if (uri == NULL || session_id_out == NULL) {
        return -1;
    }
    
    if (session_count >= MAX_SESSIONS) {
        return -1;  // No available slots
    }
    
    // Find empty slot
    int slot = -1;
    for (int i = 0; i < MAX_SESSIONS; i++) {
        if (sessions[i].session_id[0] == '\0') {
            slot = i;
            break;
        }
    }
    
    if (slot < 0) {
        return -1;
    }
    
    // Initialize session
    rtsp_session_t* session = &sessions[slot];
    generate_session_id(session->session_id);
    session->state = SESSION_STATE_INIT;
    strncpy(session->uri, uri, sizeof(session->uri) - 1);
    session->uri[sizeof(session->uri) - 1] = '\0';
    session->created_at = time(NULL);
    session->last_activity = session->created_at;
    session->transport_port = transport_port;
    
    // Copy session ID to output
    strcpy(session_id_out, session->session_id);
    
    session_count++;
    return 0;
}

/**
 * Find a session by ID.
 * 
 * @param session_id  Session ID to find
 * @return Pointer to session, or NULL if not found
 */
rtsp_session_t* find_session(const char* session_id) {
    if (session_id == NULL) {
        return NULL;
    }
    
    for (int i = 0; i < MAX_SESSIONS; i++) {
        if (strcmp(sessions[i].session_id, session_id) == 0) {
            return &sessions[i];
        }
    }
    
    return NULL;
}

/**
 * Update session state.
 * 
 * @param session_id  Session ID
 * @param new_state   New state
 * @return 0 on success, -1 on error
 */
int update_session_state(const char* session_id, session_state_t new_state) {
    rtsp_session_t* session = find_session(session_id);
    if (session == NULL) {
        return -1;
    }
    
    session->state = new_state;
    session->last_activity = time(NULL);
    
    return 0;
}

/**
 * Destroy a session.
 * 
 * @param session_id  Session ID to destroy
 * @return 0 on success, -1 on error
 */
int destroy_session(const char* session_id) {
    rtsp_session_t* session = find_session(session_id);
    if (session == NULL) {
        return -1;
    }
    
    memset(session, 0, sizeof(*session));
    session_count--;
    
    return 0;
}

/**
 * Get session timeout status.
 * 
 * @param session_id   Session ID
 * @param timeout_sec  Timeout in seconds
 * @return 1 if timed out, 0 if active, -1 on error
 */
int is_session_timed_out(const char* session_id, int timeout_sec) {
    rtsp_session_t* session = find_session(session_id);
    if (session == NULL) {
        return -1;
    }
    
    time_t now = time(NULL);
    return (now - session->last_activity) > timeout_sec ? 1 : 0;
}

