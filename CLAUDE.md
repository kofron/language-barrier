# Overall Important Guidelines
- You have access to git, and you can use it liberally to see diffs and understand where you're at relative to main.
- Your code should be legible and well-commented.  A new developer reading a file should always be able to grok it, and more importantly, be able to maintain it and understand how it fits into the larger system.
- The best possible documentation is really doctests!  Always use them whenever possible.
- Always always ALWAYS seek to extend what exists.  Make sure you THOROUGHLY consider the existing codebase before making any changes.  You should definitely consult DESIGN.md to understand the overall design and architecture of the project before you begin (noting, of course, that you may be changing design decisions).
- When you make design decisions or update existing design decisions, write it down in DESIGN.md in a "diary entry" style.  Don't delete old entries, just append.
- Unless the goal is explicitly to create a stub, you must implement the code fully.  All tests must pass and all functionality must be in place.  NO cheating.
- Similarly, no commenting out failing tests.  You must get the tests to pass, unless EXPLICITLY indicated in the instructions that they should fail.
- Whenever you think you're done, run cargo clippy to find out if you've left a mess.  Clean up after yourself if you did.
