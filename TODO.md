# Core
- [ ] Gemini function declaration types are a little funky.
- [ ] There's an interaction between provider/tool that we need to consider.  In particular, the json schema parameters actually ARENT consistent across the board - goog is special (of course) and so they follow a slightly different format.  This can actually be dealt with in consuming crates by implementing schema() for the tools in question, but it's a little annoying and feels like we could probably deal with it here while still leaving the customization knobs intact.
- [x] Need to support tool_choice.
---- HIGH IMPORTANCE ----
- [ ] Need more unit tests for the actual generated schemas that are sent to providers.
- [ ] More model config parameters around thinking
