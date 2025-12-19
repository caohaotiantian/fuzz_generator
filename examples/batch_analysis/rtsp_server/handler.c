/**
 * RTSP Request Handler
 * 
 * Simulated RTSP server request handling for fuzz testing demonstration.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_REQUEST_SIZE 4096
#define MAX_METHOD_LEN 16
#define MAX_URI_LEN 512
#define MAX_VERSION_LEN 16

/**
 * RTSP request structure
 */
typedef struct {
    char method[MAX_METHOD_LEN];
    char uri[MAX_URI_LEN];
    char version[MAX_VERSION_LEN];
    int cseq;
    char session_id[64];
} rtsp_request_t;

/**
 * Handle an incoming RTSP request.
 * 
 * @param request_data  Raw request data
 * @param data_len      Length of request data
 * @param response      Output buffer for response
 * @param response_len  Size of response buffer
 * @return Response length on success, -1 on error
 */
int handle_request(const char* request_data, int data_len,
                   char* response, int response_len) {
    rtsp_request_t request;
    
    if (request_data == NULL || data_len <= 0) {
        return -1;
    }
    
    if (response == NULL || response_len <= 0) {
        return -1;
    }
    
    // Parse the request
    memset(&request, 0, sizeof(request));
    
    // Extract method (first word)
    const char* space = strchr(request_data, ' ');
    if (space == NULL || (space - request_data) >= MAX_METHOD_LEN) {
        snprintf(response, response_len, "RTSP/1.0 400 Bad Request\r\n\r\n");
        return strlen(response);
    }
    
    strncpy(request.method, request_data, space - request_data);
    
    // Route to appropriate handler
    if (strcmp(request.method, "OPTIONS") == 0) {
        snprintf(response, response_len,
                 "RTSP/1.0 200 OK\r\n"
                 "CSeq: %d\r\n"
                 "Public: OPTIONS, DESCRIBE, SETUP, PLAY, TEARDOWN\r\n\r\n",
                 request.cseq);
    } else if (strcmp(request.method, "DESCRIBE") == 0) {
        snprintf(response, response_len,
                 "RTSP/1.0 200 OK\r\n"
                 "CSeq: %d\r\n"
                 "Content-Type: application/sdp\r\n\r\n",
                 request.cseq);
    } else {
        snprintf(response, response_len,
                 "RTSP/1.0 501 Not Implemented\r\n"
                 "CSeq: %d\r\n\r\n",
                 request.cseq);
    }
    
    return strlen(response);
}

/**
 * Handle OPTIONS request.
 * 
 * @param cseq      Command sequence number
 * @param response  Output buffer
 * @param len       Buffer size
 * @return Response length
 */
int handle_options(int cseq, char* response, int len) {
    if (response == NULL || len <= 0) {
        return -1;
    }
    
    return snprintf(response, len,
                    "RTSP/1.0 200 OK\r\n"
                    "CSeq: %d\r\n"
                    "Public: OPTIONS, DESCRIBE, SETUP, PLAY, PAUSE, TEARDOWN\r\n\r\n",
                    cseq);
}

