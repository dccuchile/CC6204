from flask.json import jsonify


def error(message, code):
    return jsonify({
        'status': 'error',
        'code': code,
        'message': message

    })


def send_test(test_data):
    return jsonify({
        'status': 'OK',
        'data': test_data
    })


def send_results(status, mask, comments):
    return jsonify({
        'status': 'OK',
        'result_status': status,
        'mask': mask,
        'comments': comments
    })
