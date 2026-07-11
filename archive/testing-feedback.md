## Overall
- Score breakdown is a bit too black box still from within the interface, especially taste profiling and list signals
- Additionally it seems that list signals are too simplistic, just because a movie is in a couple of list does not mean that the movie should be a top candidate

- Need to find a balance between discovering new movies and really evaluating watchlisted ones

- "Taste mode" side bar feature only seems to impact match scoring and not 

- I want to integrate watch counts into my app somewhere (Movies that I have watched a lot already should be taken into account somehow in my taste profiling)

- Explore minimal but fun and additive LLM integration (manual triggers needed)

Your (Claude Code's) feedback:
1. Use the feedback system you already built. Wire the "Tune watched movies" UI from todo.md and start tagging real films. Until feedback has rows, half your scoring pipeline (add_feedback_similarity) is dead weight you can't tune.
2. Establish a quality signal. Even something lightweight: pick 20 watched-but-"recommendable" films, see where the recommender ranks them, write the result into that empty testing-feedback.md. Right now you have no ground truth, so you can't know if a feature helped.


## Add a taste profile page
- Really go in depth on how my taste is situated, how it has been developping


### Tonights Pick:
- In the mood for does not have sufficient effect on the recommendations, there is too much overlap in recommendations between the movies recommended 
- Standard recommendation does not change (unless it does not fit the time available requirement): For example Recommended Chinatown, even when changed to in the mood for comfort movie and avoid these moods: "Gritty"
- "Appears in list" as way too much influence

### Recommendations
- When choosing a film to anchor on: Can I only choose movies I have already watched? 
        If for example I want to find recommendations based on the upcoming movie "The Odysessy" how should I find recommendations for that? Is this more something for the curated weeks feature?
- Filders > Mood is unselectable
- The language feature does not seem to be working correctly
- The scoring seems off: Whenever I adapt the scoring weights my scores of all the recommendations changes scores but the ranking or selection barely changes , the results also dont really makes sense

For example:
When I select the following
Taste mode = Comfort Movie
Anchor Film: "In the Mood for Love" (2000)
Not in the mood tonight for: Gritty, Tense
No filters selected

Scoring weights: 
Taste similarity  = 3
Theme similarity = 1
Director/Cast influence = 3
List signals = 0
Anchor influence = 1
Watched movies feedback = 1


My top 4 selection (and match score) then is:
- The long goodbye (12.21)
- Misery (9.91)
- Touch of Evil (8.45)
- Lost Highway (7.25)


This top 4 for is identical when I select the standard weights (all =1) , taste mode = balanced, and no anchor focus or moods to avoid selected, just with diffrent match scores:
- The long goodbye (11.63)
- Misery (8.94)
- Touch of Evil (7.93)
- Lost Highway (7.78)




### Analysis Page
- Im not happy yet with the Tune watched movies page yet a couple of reasons:
        - Visualisation of the movies is still a bit clunky and too random, movies are now order alphabetical with no way to edit the list execpt for the search featur
        - Taste feedback options still do not cover how I want to evaluate a movie
        - Adding review data (If already available from letterboxd) for future LLM usage would be a fun addition
        Seeing the effects of the feedback would make the tuning watched movie feature more interactive
        - The graphs are visually jarring (coloring, too basic of a setup both from a visual as well as substantive level), Considering if these visuals should live separate from the tune watched movies function ( and maybe move it to a "My taste profile overview"-page)

### Curated Weeks:
- The selection of movies seems too random still and genre overlap seems to be too dominant of a factor (f.e. Mission Impossible recommendations when selecting Inception as a movie anchor)

- When I try "Director Focused" For Robert Eggers'Nosferatu there is some Good recommendations, but it still needs some work
Here is some concrete feedback
Day 1: Werefulf (2026); Has not come out yet, although I like the recommend, I would actually like to select this as an anchor movie instead, but it is not selectable. Additionally since its the newest movie, it seems like a bad selection to select this as the first movie of the curated week
- Day 2: The Lighthouse (2019): Great recommend, no notes, except for maybe the order? 

- The northman (2022): Very logical recommend, have already seen it, so would love to have the option to replace it, get a new recommend to optionally select?

- Day 4: Anchor movie itself, again only note is the possibility to maybe change the order

Day 5: The excorcist (1973) Seems like a logical recommended, although its not from the same director obviously, so if the excorcist has the same "lineage", this would be a great pick if put earlier

Day 6: Obsession (2026) I really like the pick, again is it the most relevant in terms of "Director focus" instead of theme focus? Maybe not, nonetheless a good pick even though it only just came out

Day 7: Twilight (2008) A very funny pick, but idk if this fits the purpose of the feature it is a nice distraction while staying in theme of the anchor movie, but it distracts from the theme focus and the more serious "Movie curation" narrative of the feature. Nonetheless I dont dislike the selection.


Overall it seems like the biggest gap is actually the missing of additional movie data, for example when you at the youtube video of the
"The 'Nosferatu' Syllabus | The Big Picture" you can see the host selects "The Witch (2015)" from Eggers as a logical 'Director focused pick' in his syllabus. This should be an obvious selection in the Movie Curation feature with the selected parameters as well. I think the main reason for this is that my database is simply "Missing" any data about the movie




-- The Odysessy Curation: Would it be possible for each recommend, to change the order, and make it re-attempt individually (based on full recommended list and specific feedback), save the retry as a feedback signal that can be used to refine later


