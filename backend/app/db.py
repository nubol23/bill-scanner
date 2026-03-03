from __future__ import annotations

from psycopg_pool import AsyncConnectionPool


def create_async_pool(database_url: str) -> AsyncConnectionPool:
  return AsyncConnectionPool(
    conninfo=database_url,
    min_size=1,
    max_size=4,
    open=False,
    kwargs={"autocommit": True},
  )
