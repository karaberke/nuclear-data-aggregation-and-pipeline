"""Framework-independent application services.

Every function here is callable from both a FastAPI route and a Dash
callback, and imports neither. This is the seam that keeps the Dash UI from
having to talk HTTP to its own process.
"""
