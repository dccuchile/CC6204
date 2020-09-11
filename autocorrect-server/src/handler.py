from flask.json import jsonify


def error(message, code):
    return jsonify({
        "error": {
            "code": code,
            "text": message
        }
    })


def send_test(test_data):
    return jsonify({
        "data": test_data
    })


def send_results(status, mask, comments):
    return jsonify({
        "status": status,
        "mask": mask,
        "comments": comments
    })
