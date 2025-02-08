
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def custom_openapi(app: FastAPI):
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Custom FastAPI",
        version="1.0.0",
        description=(
            "This is a custom OpenAPI schema with authentication applied"
        ),
        routes=app.routes,
    )

    if "securitySchemes" not in openapi_schema["components"]:
        openapi_schema["components"]["securitySchemes"] = {}

    openapi_schema["components"]["securitySchemes"]["BearerAuth"] = {
        "type": "http",
        "scheme": "Bearer",
        "bearerFormat": "",
        "description": "Enter your BearerAuth token to authenticate"
    }

    for _, methods in openapi_schema["paths"].items():
        for method in methods:
            methods[method]["security"] = [{"BearerAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema
