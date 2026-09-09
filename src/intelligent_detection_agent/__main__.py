import uvicorn


def main() -> None:
    """启动本地 FastAPI 服务。"""

    uvicorn.run(
        "intelligent_detection_agent.api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
