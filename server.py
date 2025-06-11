# -*- coding: utf-8 -*-

import os
import argparse
from collections import OrderedDict

from flask import Flask, request, jsonify
from intent_classifier import IntentClassifier

# Global model instance
app = Flask(__name__)
model = IntentClassifier()


@app.route('/ready', methods=['GET'])
def ready():
    """Health check endpoint"""
    if model.is_ready():
        return 'OK', 200
    else:
        return 'Not ready', 423


@app.route('/intent', methods=['POST'])
def intent():
    """Intent classification endpoint"""
    import json
    from flask import Response

    try:
        # Validate request content type
        if not request.is_json:
            return jsonify({
                "label": "INVALID_CONTENT_TYPE",
                "message": "Content-Type must be application/json"
            }), 400

        # Get request data
        data = request.get_json()

        # Validate required field exists
        if 'text' not in data:
            return jsonify({
                "label": "MISSING_TEXT_FIELD",
                "message": "\"text\" field is required"
            }), 400

        text = data['text']

        # Validate text is not empty
        if not text or text.strip() == '':
            return jsonify({
                "label": "TEXT_EMPTY",
                "message": "\"text\" is empty."
            }), 400

        # Check if model is ready
        if not model.is_ready():
            return jsonify({
                "label": "MODEL_NOT_READY",
                "message": "Model is not loaded or ready"
            }), 503

        # Get predictions from model
        predictions = model.predict(text.strip())

        # Format response with guaranteed order - manual JSON construction
        intents_json_parts = []
        for pred in predictions[:3]:
            intent_json = f'{{"label":"{pred["label"]}","confidence":{round(pred["confidence"], 3)}}}'
            intents_json_parts.append(intent_json)

        # Construct final JSON manually to guarantee order
        response_json = f'{{"intents":[{",".join(intents_json_parts)}]}}'

        return Response(
            response_json,
            mimetype='application/json',
            status=200
        )

    except Exception as e:
        # Handle any unexpected errors
        return jsonify({
            "label": "INTERNAL_ERROR",
            "message": str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "label": "NOT_FOUND",
        "message": "Endpoint not found"
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "label": "METHOD_NOT_ALLOWED",
        "message": "Method not allowed for this endpoint"
    }), 405


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model', type=str, required=True, help='Path to model directory.')
    arg_parser.add_argument('--port', type=int, default=int(os.getenv('PORT', 8080)), help='Server port number.')
    args = arg_parser.parse_args()

    # FIX THE BUG: Load model BEFORE starting server
    print(f"Loading model from: {args.model}")
    try:
        model.load(args.model)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1

    # Start the Flask server
    print(f"🚀 Starting server on port {args.port}")
    print(f"🔗 Health check: http://localhost:{args.port}/ready")
    print(f"🎯 Classification: http://localhost:{args.port}/intent")

    app.run(host='0.0.0.0', port=args.port, debug=False)
    return 0


if __name__ == '__main__':
    main()
