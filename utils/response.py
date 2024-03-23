from flask import Response, json


def success_dto(message="OK", data=None, meta=None):
    res = {
        "status": True,
        "message": message,
    }
    if data:
        res["data"] = data
    if meta:
        res["meta"] = meta
    return json.dumps(res)


def error_dto(message="Error", errors=None):
    res = {
        "status": False,
        "message": message,
    }
    if errors:
        res["errors"] = errors
    return json.dumps(res)


def success(message="OK", data=None, meta=None):
    return Response(
        status=200,
        response=success_dto(message, data, meta),
        mimetype='application/json'
    )


def bad_request(message="Bad Request!", errors=None):
    return Response(
        status=400,
        response=error_dto(message, errors),
        mimetype='application/json'
    )


def internal_server_error(message="Internal Server Error!"):
    return Response(
        status=500,
        response=error_dto(message),
        mimetype='application/json'
    )
