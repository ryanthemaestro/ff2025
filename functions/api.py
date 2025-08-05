import json
import os
import sys
from urllib.parse import unquote

# Add the scripts directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

# Import the Flask app from our draft_ui module
from draft_ui import app

def handler(event, context):
    """
    AWS Lambda handler function for Netlify Functions
    """
    try:
        # Get HTTP method and path
        http_method = event.get('httpMethod', 'GET')
        path = event.get('path', '/')
        
        # Remove /api prefix if present
        if path.startswith('/api'):
            path = path[4:]
        if not path:
            path = '/'
            
        # Get query string parameters
        query_string_parameters = event.get('queryStringParameters') or {}
        
        # Get headers
        headers = event.get('headers', {})
        
        # Get body
        body = event.get('body', '')
        if event.get('isBase64Encoded', False):
            import base64
            body = base64.b64decode(body).decode('utf-8')
        
        # Create a test request context
        with app.test_request_context(
            path=path, 
            method=http_method, 
            query_string='&'.join([f"{k}={v}" for k, v in query_string_parameters.items()]),
            headers=headers,
            data=body
        ):
            # Process the request
            response = app.full_dispatch_request()
            
            # Get response data
            response_data = response.get_data(as_text=True)
            
            # Return the response in Netlify Functions format
            return {
                'statusCode': response.status_code,
                'headers': {
                    'Content-Type': response.content_type,
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS'
                },
                'body': response_data
            }
            
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': f'Internal server error: {str(e)}'})
        } 