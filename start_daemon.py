"""Simple script to start Reachy Mini daemon in simulation mode on Windows."""
import asyncio
import logging

logging.basicConfig(level=logging.INFO)

async def main():
    from reachy_mini.daemon.daemon import Daemon

    daemon = Daemon(log_level="INFO")
    await daemon.run4ever(sim=False, headless=True)

if __name__ == "__main__":
    asyncio.run(main())
