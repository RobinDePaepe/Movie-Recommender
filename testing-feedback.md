## Overall
- Score breakdown is a bit too black box still from within the interface, especially taste profiling and list signals
- Additionally it seems that list signals are too simplistic, just because a movie is in a couple of list does not mean that the movie should be a top candidate

- Need to find a balance between discovering new movies and really evaluating watchlisted ones

- I want to integrate watch counts into my app somewhere (Movies that I have watched a lot already should be taken into account somehow in my taste profiling)

- Explore minimal but fun and additive LLM integration (manual triggers needed)

Your feedback:
1. Use the feedback system you already built. Wire the "Tune watched movies" UI from todo.md and start tagging real films. Until feedback has rows, half your scoring pipeline (add_feedback_similarity) is dead weight you can't tune.
2. Establish a quality signal. Even something lightweight: pick 20 watched-but-"recommendable" films, see where the recommender ranks them, write the result into that empty testing-feedback.md. Right now you have no ground truth, so you can't know if a feature helped.


## Add a taste profile page
- Really go in depth on how my taste is situated, how it has been developping


### Tonights Pick:
- In the mood for does not have sufficient effect on the recommendations, there is too much overlap in recommendations between the movies recommended 

### Analysis Page
- Im not happy yet with the Tune watched movies page yet a couple of reasons:
        - Visualisation of the movies is still a bit clunky and too random, movies are now order alphabetical with no way to edit the list execpt for the search featur
        - Taste feedback options still do not cover how I want to evaluate a movie
        - Adding review data (If already available from letterboxd) for future LLM usage would be a fun addityion
        Seeing the effects of the feedback would make the tuning watched movie feature more interactive
        - The graphs are visually jarring (coloring, too basic of a setup both from a visual as well as substantive level), Considering if these visuals should live separate from the tune watched movies function ( and maybe move it to a "My taste profile overview"-page)

### Curated Weeks:
- The selection of movies seems too random still and genre overlap seems to be too dominant of a factor (f.e. Mission Impossible recommendations when selecting Inception as a movie anchor)


