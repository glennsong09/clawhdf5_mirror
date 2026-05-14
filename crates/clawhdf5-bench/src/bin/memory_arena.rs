//! MemoryArena-style Multi-Session Benchmark (Track 8.2)
//!
//! Generates deterministic synthetic multi-session conversations and evaluates
//! the retrieval system's ability to answer single-session and cross-session queries.
//!
//! # Design
//! - 50 sessions × 20 turns = 1000 total turns per "user"
//! - Topics: personal info, preferences, past events, schedule items, hobbies
//! - 100 queries: single-session, cross-session, temporal ordering, contradiction
//! - All embeddings are zero vectors; evaluation is BM25-only
//!
//! # Reproducibility
//! All data is deterministic — no random number generator is used.
//!
//! # Usage
//! ```
//! cargo run --release --bin memory_arena
//! ```

use std::collections::HashSet;
use std::time::{Duration, Instant};

use clawhdf5_agent::{AgentMemory, HDF5Memory, MemoryConfig, MemoryEntry};
use tempfile::TempDir;

const EMBEDDING_DIM: usize = 384;

// ---------------------------------------------------------------------------
// Synthetic session data
// ---------------------------------------------------------------------------

/// A single conversation turn.
struct Turn {
    #[allow(dead_code)]
    role: &'static str,
    content: &'static str,
}

/// A conversation session.
struct Session {
    id: &'static str,
    turns: &'static [Turn],
}

/// A benchmark query with ground-truth session IDs.
struct Query {
    question: &'static str,
    /// Session IDs whose content answers this question.
    expected_sessions: &'static [&'static str],
    query_type: &'static str,
}

// ---------------------------------------------------------------------------
// Static session definitions  (deterministic seed data)
// ---------------------------------------------------------------------------

const SESSIONS: &[Session] = &[
    // -- PERSONAL INFO (sessions 0-9) ---------------------------------------
    Session {
        id: "personal_0",
        turns: &[
            Turn {
                role: "user",
                content: "Hi, my full name is Alexander Mitchell.",
            },
            Turn {
                role: "assistant",
                content: "Nice to meet you, Alexander!",
            },
            Turn {
                role: "user",
                content: "I'm 34 years old and I was born in Boston.",
            },
            Turn {
                role: "assistant",
                content: "Boston is a great city.",
            },
            Turn {
                role: "user",
                content: "I currently live in Portland Oregon with my wife Emma.",
            },
            Turn {
                role: "assistant",
                content: "Portland is lovely.",
            },
            Turn {
                role: "user",
                content: "Emma and I have a daughter named Sofia, she is 5.",
            },
            Turn {
                role: "assistant",
                content: "That's wonderful!",
            },
            Turn {
                role: "user",
                content: "We also have a dog named Biscuit, a golden retriever.",
            },
            Turn {
                role: "assistant",
                content: "Golden retrievers are such friendly dogs.",
            },
            Turn {
                role: "user",
                content: "My mother's name is Margaret and she lives in Seattle.",
            },
            Turn {
                role: "assistant",
                content: "Do you visit her often?",
            },
            Turn {
                role: "user",
                content: "Yes, we drive up to Seattle about once a month.",
            },
            Turn {
                role: "assistant",
                content: "That's nice that you stay close.",
            },
            Turn {
                role: "user",
                content: "My brother Daniel works as a paramedic in Denver.",
            },
            Turn {
                role: "assistant",
                content: "That's an important profession.",
            },
            Turn {
                role: "user",
                content: "We grew up in the same neighborhood in Boston's South End.",
            },
            Turn {
                role: "assistant",
                content: "The South End is a great neighborhood.",
            },
            Turn {
                role: "user",
                content: "My email is alex.mitchell@protonmail.com if you need it.",
            },
            Turn {
                role: "assistant",
                content: "Got it, I'll remember that.",
            },
        ],
    },
    Session {
        id: "personal_1",
        turns: &[
            Turn {
                role: "user",
                content: "I work as a senior software architect at QuantumLogic Industries.",
            },
            Turn {
                role: "assistant",
                content: "That sounds like a challenging role.",
            },
            Turn {
                role: "user",
                content: "QuantumLogic is headquartered in downtown Portland on SW 5th Avenue.",
            },
            Turn {
                role: "assistant",
                content: "Convenient location.",
            },
            Turn {
                role: "user",
                content: "I've been with the company for seven years now.",
            },
            Turn {
                role: "assistant",
                content: "Seven years is a solid tenure.",
            },
            Turn {
                role: "user",
                content: "My manager is Dr. Patricia Huang, VP of Engineering.",
            },
            Turn {
                role: "assistant",
                content: "What does your team work on?",
            },
            Turn {
                role: "user",
                content: "We build distributed systems and real-time data pipelines in Rust and Go.",
            },
            Turn {
                role: "assistant",
                content: "Rust and Go are both great for systems work.",
            },
            Turn {
                role: "user",
                content: "My annual salary is around 185000 dollars.",
            },
            Turn {
                role: "assistant",
                content: "That's a strong compensation.",
            },
            Turn {
                role: "user",
                content: "I get 20 days of paid vacation per year plus federal holidays.",
            },
            Turn {
                role: "assistant",
                content: "That's a reasonable benefits package.",
            },
            Turn {
                role: "user",
                content: "My employee ID at QuantumLogic is QLE-40271.",
            },
            Turn {
                role: "assistant",
                content: "I'll keep that on record.",
            },
            Turn {
                role: "user",
                content: "I hold a Master's degree in Computer Science from MIT.",
            },
            Turn {
                role: "assistant",
                content: "Impressive academic background.",
            },
            Turn {
                role: "user",
                content: "My undergraduate degree was in Electrical Engineering from Purdue.",
            },
            Turn {
                role: "assistant",
                content: "A strong technical foundation.",
            },
        ],
    },
    // -- FOOD PREFERENCES (sessions 2-3) ------------------------------------
    Session {
        id: "prefs_food_0",
        turns: &[
            Turn {
                role: "user",
                content: "My absolute favorite cuisine is Thai food.",
            },
            Turn {
                role: "assistant",
                content: "Thai food is delicious!",
            },
            Turn {
                role: "user",
                content: "Specifically I love pad see ew with extra wide noodles.",
            },
            Turn {
                role: "assistant",
                content: "That's a great dish.",
            },
            Turn {
                role: "user",
                content: "I also enjoy green curry with tofu, not chicken.",
            },
            Turn {
                role: "assistant",
                content: "Tofu absorbs the curry flavor well.",
            },
            Turn {
                role: "user",
                content: "My favorite Thai restaurant in Portland is Pok Pok on SE Division.",
            },
            Turn {
                role: "assistant",
                content: "Pok Pok is very well known.",
            },
            Turn {
                role: "user",
                content: "I'm lactose intolerant so I avoid dairy products.",
            },
            Turn {
                role: "assistant",
                content: "That's important to keep in mind.",
            },
            Turn {
                role: "user",
                content: "I'm not vegetarian but I prefer plant-based meals during weekdays.",
            },
            Turn {
                role: "assistant",
                content: "That's a healthy approach.",
            },
            Turn {
                role: "user",
                content: "I also love Ethiopian injera with misir wot lentil stew.",
            },
            Turn {
                role: "assistant",
                content: "Ethiopian cuisine is very flavorful.",
            },
            Turn {
                role: "user",
                content: "For breakfast I always have oat milk coffee and avocado toast.",
            },
            Turn {
                role: "assistant",
                content: "A classic Portland breakfast.",
            },
            Turn {
                role: "user",
                content: "I try to avoid processed sugar and prefer dark chocolate.",
            },
            Turn {
                role: "assistant",
                content: "Dark chocolate has some health benefits.",
            },
            Turn {
                role: "user",
                content: "My wife Emma is a great cook and makes Thai food at home too.",
            },
            Turn {
                role: "assistant",
                content: "How lovely that you share that interest.",
            },
        ],
    },
    Session {
        id: "prefs_food_1",
        turns: &[
            Turn {
                role: "user",
                content: "I updated my food preferences — I recently started eating sushi.",
            },
            Turn {
                role: "assistant",
                content: "Sushi is a great addition.",
            },
            Turn {
                role: "user",
                content: "My favorite sushi spot is Umi Sake House in Portland.",
            },
            Turn {
                role: "assistant",
                content: "Great choice.",
            },
            Turn {
                role: "user",
                content: "I still love Thai food the most though.",
            },
            Turn {
                role: "assistant",
                content: "Some things are timeless.",
            },
            Turn {
                role: "user",
                content: "I've also been experimenting with Korean BBQ.",
            },
            Turn {
                role: "assistant",
                content: "Korean BBQ is a fun social dining experience.",
            },
            Turn {
                role: "user",
                content: "For work lunches I usually go to the food carts on SW 10th.",
            },
            Turn {
                role: "assistant",
                content: "Portland's food cart scene is legendary.",
            },
            Turn {
                role: "user",
                content: "I gave up red meat about two years ago.",
            },
            Turn {
                role: "assistant",
                content: "That's a significant dietary change.",
            },
            Turn {
                role: "user",
                content: "Now I mainly eat fish, poultry, and plant-based proteins.",
            },
            Turn {
                role: "assistant",
                content: "A balanced pescatarian-ish diet.",
            },
            Turn {
                role: "user",
                content: "I make a really good lentil soup recipe I got from my grandmother.",
            },
            Turn {
                role: "assistant",
                content: "Family recipes are special.",
            },
            Turn {
                role: "user",
                content: "I drink about four cups of coffee a day, always black.",
            },
            Turn {
                role: "assistant",
                content: "That's quite a lot of coffee.",
            },
            Turn {
                role: "user",
                content: "Emma prefers herbal teas over coffee.",
            },
            Turn {
                role: "assistant",
                content: "Opposite tastes in beverages.",
            },
        ],
    },
    // -- MUSIC PREFERENCES (session 4) -------------------------------------
    Session {
        id: "prefs_music_0",
        turns: &[
            Turn {
                role: "user",
                content: "I'm a huge jazz fan — especially Miles Davis and John Coltrane.",
            },
            Turn {
                role: "assistant",
                content: "Two of the greatest jazz musicians.",
            },
            Turn {
                role: "user",
                content: "My all-time favorite album is Kind of Blue by Miles Davis.",
            },
            Turn {
                role: "assistant",
                content: "A classic, considered one of the best jazz albums ever.",
            },
            Turn {
                role: "user",
                content: "I also listen to a lot of classical music, especially Debussy.",
            },
            Turn {
                role: "assistant",
                content: "Debussy's impressionist style pairs well with jazz appreciation.",
            },
            Turn {
                role: "user",
                content: "I play the piano — I've been playing since I was eight years old.",
            },
            Turn {
                role: "assistant",
                content: "That's over 25 years of practice.",
            },
            Turn {
                role: "user",
                content: "I practice piano every morning for about 30 minutes before work.",
            },
            Turn {
                role: "assistant",
                content: "A disciplined routine.",
            },
            Turn {
                role: "user",
                content: "I have a Steinway Model B grand piano in the living room.",
            },
            Turn {
                role: "assistant",
                content: "A Steinway Model B is a beautiful instrument.",
            },
            Turn {
                role: "user",
                content: "I also enjoy ambient electronic music for focus work — Brian Eno.",
            },
            Turn {
                role: "assistant",
                content: "Brian Eno is perfect for deep work.",
            },
            Turn {
                role: "user",
                content: "My Spotify username is alex.mitchell.pdx if you want to see my playlists.",
            },
            Turn {
                role: "assistant",
                content: "I'll note that.",
            },
            Turn {
                role: "user",
                content: "I don't really like country music or heavy metal.",
            },
            Turn {
                role: "assistant",
                content: "Everyone has their preferences.",
            },
            Turn {
                role: "user",
                content: "Sofia is starting piano lessons next month at age five.",
            },
            Turn {
                role: "assistant",
                content: "Starting young is wonderful for musical development.",
            },
        ],
    },
    // -- TRAVEL (sessions 5-6) ---------------------------------------------
    Session {
        id: "travel_0",
        turns: &[
            Turn {
                role: "user",
                content: "I went to Tokyo Japan last March for two weeks.",
            },
            Turn {
                role: "assistant",
                content: "Tokyo is an amazing city.",
            },
            Turn {
                role: "user",
                content: "It was my first time visiting Japan and I was blown away by everything.",
            },
            Turn {
                role: "assistant",
                content: "Japan is known for making strong first impressions.",
            },
            Turn {
                role: "user",
                content: "I stayed at the Park Hyatt Tokyo in Shinjuku, the hotel from Lost in Translation.",
            },
            Turn {
                role: "assistant",
                content: "That's a legendary hotel with iconic views.",
            },
            Turn {
                role: "user",
                content: "I ate ramen every single day — specifically tonkotsu style.",
            },
            Turn {
                role: "assistant",
                content: "Tokyo ramen is world class.",
            },
            Turn {
                role: "user",
                content: "I visited the Tsukiji outer market at 5am for fresh sushi breakfast.",
            },
            Turn {
                role: "assistant",
                content: "That's a classic Tokyo experience.",
            },
            Turn {
                role: "user",
                content: "I also did a day trip to Kyoto and saw the Fushimi Inari shrine.",
            },
            Turn {
                role: "assistant",
                content: "The thousand torii gates are breathtaking.",
            },
            Turn {
                role: "user",
                content: "Total trip cost was around 8000 dollars including flights.",
            },
            Turn {
                role: "assistant",
                content: "A reasonable budget for two weeks in Japan.",
            },
            Turn {
                role: "user",
                content: "Emma stayed home with Sofia since she was still a baby at the time.",
            },
            Turn {
                role: "assistant",
                content: "That makes sense.",
            },
            Turn {
                role: "user",
                content: "I brought back wagyu beef jerky and matcha Kit Kats as souvenirs.",
            },
            Turn {
                role: "assistant",
                content: "Great choices.",
            },
            Turn {
                role: "user",
                content: "I want to go back to Japan with the whole family in two years.",
            },
            Turn {
                role: "assistant",
                content: "Sofia would love it by then.",
            },
        ],
    },
    Session {
        id: "travel_1",
        turns: &[
            Turn {
                role: "user",
                content: "This past summer we drove along the Oregon coast for a week.",
            },
            Turn {
                role: "assistant",
                content: "The Oregon coast is beautiful.",
            },
            Turn {
                role: "user",
                content: "We stayed at Cannon Beach and at Astoria.",
            },
            Turn {
                role: "assistant",
                content: "Two great stops on the coast.",
            },
            Turn {
                role: "user",
                content: "Sofia loved the tide pools at Cannon Beach.",
            },
            Turn {
                role: "assistant",
                content: "Tide pools are endlessly fascinating for kids.",
            },
            Turn {
                role: "user",
                content: "Biscuit came along on the trip, the dog loves the beach.",
            },
            Turn {
                role: "assistant",
                content: "Dogs at the beach are always happy.",
            },
            Turn {
                role: "user",
                content: "We rented a small cabin with a ocean view outside of Lincoln City.",
            },
            Turn {
                role: "assistant",
                content: "Sounds like a wonderful family trip.",
            },
            Turn {
                role: "user",
                content: "We ate at a seafood place called Pacific Halibut Co. in Newport.",
            },
            Turn {
                role: "assistant",
                content: "Fresh Pacific seafood is hard to beat.",
            },
            Turn {
                role: "user",
                content: "It rained three of the seven days but that's Oregon for you.",
            },
            Turn {
                role: "assistant",
                content: "Rain is part of the Oregon coastal experience.",
            },
            Turn {
                role: "user",
                content: "We're planning to do California next summer — Highway 1 road trip.",
            },
            Turn {
                role: "assistant",
                content: "Highway 1 is one of the most scenic drives in the world.",
            },
            Turn {
                role: "user",
                content: "I want to stop at Big Sur and see the elephant seals at San Simeon.",
            },
            Turn {
                role: "assistant",
                content: "The elephant seal rookery at San Simeon is spectacular.",
            },
            Turn {
                role: "user",
                content: "Emma wants to visit the wine country in Sonoma too.",
            },
            Turn {
                role: "assistant",
                content: "Sonoma is lovely.",
            },
        ],
    },
    // -- WORK/SCHEDULE (sessions 7-8) --------------------------------------
    Session {
        id: "work_0",
        turns: &[
            Turn {
                role: "user",
                content: "The Prometheus project deadline is October 15th.",
            },
            Turn {
                role: "assistant",
                content: "Got it, October 15th for the Prometheus project.",
            },
            Turn {
                role: "user",
                content: "Prometheus is our real-time event streaming platform rewrite in Rust.",
            },
            Turn {
                role: "assistant",
                content: "That sounds like a major initiative.",
            },
            Turn {
                role: "user",
                content: "We have a team of six engineers on Prometheus.",
            },
            Turn {
                role: "assistant",
                content: "A focused team.",
            },
            Turn {
                role: "user",
                content: "The main technical challenge is achieving sub-millisecond P99 latency.",
            },
            Turn {
                role: "assistant",
                content: "That's an ambitious latency target.",
            },
            Turn {
                role: "user",
                content: "My weekly one-on-one with Dr. Huang is every Tuesday at 10am.",
            },
            Turn {
                role: "assistant",
                content: "I'll note that recurring meeting.",
            },
            Turn {
                role: "user",
                content: "I have a board presentation scheduled for September 28th at 2pm.",
            },
            Turn {
                role: "assistant",
                content: "Important presentation — shall I help you prepare?",
            },
            Turn {
                role: "user",
                content: "The standup with the Prometheus team is every weekday at 9:15am.",
            },
            Turn {
                role: "assistant",
                content: "Early morning standups keep the team aligned.",
            },
            Turn {
                role: "user",
                content: "My direct reports are Kenji Tanaka, Maria Santos, and Liam O'Brien.",
            },
            Turn {
                role: "assistant",
                content: "A small but likely effective team.",
            },
            Turn {
                role: "user",
                content: "Maria is leading the Kafka integration component.",
            },
            Turn {
                role: "assistant",
                content: "Kafka integration is complex but well-documented.",
            },
            Turn {
                role: "user",
                content: "Kenji handles the distributed consensus layer using Raft.",
            },
            Turn {
                role: "assistant",
                content: "Raft is a good choice for understandable consensus.",
            },
        ],
    },
    Session {
        id: "work_1",
        turns: &[
            Turn {
                role: "user",
                content: "I have a performance review scheduled for November 3rd.",
            },
            Turn {
                role: "assistant",
                content: "I'll note November 3rd for your review.",
            },
            Turn {
                role: "user",
                content: "I'm aiming for a Senior Principal promotion in this review cycle.",
            },
            Turn {
                role: "assistant",
                content: "That's a significant step up.",
            },
            Turn {
                role: "user",
                content: "The Prometheus deadline was moved to November 1st from October 15th.",
            },
            Turn {
                role: "assistant",
                content: "A 2-week extension on the Prometheus project.",
            },
            Turn {
                role: "user",
                content: "The change was because the Kafka team at our client had a delay.",
            },
            Turn {
                role: "assistant",
                content: "External dependencies often cause schedule shifts.",
            },
            Turn {
                role: "user",
                content: "I need to submit my self-evaluation by October 20th.",
            },
            Turn {
                role: "assistant",
                content: "October 20th for the self-evaluation.",
            },
            Turn {
                role: "user",
                content: "Liam O'Brien is leaving the company at the end of October.",
            },
            Turn {
                role: "assistant",
                content: "That's a significant loss for the team.",
            },
            Turn {
                role: "user",
                content: "We're interviewing to replace Liam — I have three candidate interviews next week.",
            },
            Turn {
                role: "assistant",
                content: "Good to act quickly on backfilling.",
            },
            Turn {
                role: "user",
                content: "Dr. Huang approved a 12 percent budget increase for Q1 next year.",
            },
            Turn {
                role: "assistant",
                content: "More budget means more resources.",
            },
            Turn {
                role: "user",
                content: "I'm planning to hire two additional senior engineers with that budget.",
            },
            Turn {
                role: "assistant",
                content: "Growing the team strategically.",
            },
            Turn {
                role: "user",
                content: "The board presentation went well — they approved our roadmap.",
            },
            Turn {
                role: "assistant",
                content: "Great news on the board approval.",
            },
        ],
    },
    // -- HOBBIES (sessions 9) -----------------------------------------------
    Session {
        id: "hobbies_0",
        turns: &[
            Turn {
                role: "user",
                content: "I run every morning — typically 5 kilometers around Laurelhurst Park.",
            },
            Turn {
                role: "assistant",
                content: "Laurelhurst Park is a beautiful running location.",
            },
            Turn {
                role: "user",
                content: "I've run three marathons: Portland 2019, Boston 2021, and Chicago 2023.",
            },
            Turn {
                role: "assistant",
                content: "Three marathons is impressive.",
            },
            Turn {
                role: "user",
                content: "My personal best marathon time is 3 hours 41 minutes at Boston.",
            },
            Turn {
                role: "assistant",
                content: "That's a solid time.",
            },
            Turn {
                role: "user",
                content: "I'm currently training for the Portland Marathon in October.",
            },
            Turn {
                role: "assistant",
                content: "Good luck with the training!",
            },
            Turn {
                role: "user",
                content: "I also rock climb at the Portland Rock Gym on Wednesdays.",
            },
            Turn {
                role: "assistant",
                content: "Climbing complements running nicely.",
            },
            Turn {
                role: "user",
                content: "I'm at 5.11c level on sport climbs.",
            },
            Turn {
                role: "assistant",
                content: "That's an advanced level.",
            },
            Turn {
                role: "user",
                content: "I read about two books per month — mostly sci-fi and technical books.",
            },
            Turn {
                role: "assistant",
                content: "A good mix of fiction and professional development.",
            },
            Turn {
                role: "user",
                content: "My favorite sci-fi author is Ted Chiang — Exhalation is brilliant.",
            },
            Turn {
                role: "assistant",
                content: "Ted Chiang's work is extraordinary.",
            },
            Turn {
                role: "user",
                content: "I also do woodworking in my garage on weekends.",
            },
            Turn {
                role: "assistant",
                content: "A great hands-on hobby.",
            },
            Turn {
                role: "user",
                content: "I recently built a walnut dining table that took three months to finish.",
            },
            Turn {
                role: "assistant",
                content: "A walnut dining table is a beautiful piece of furniture.",
            },
        ],
    },
    // -- FILLER SESSIONS (10-49, covering 5 additional topics with less detail)
    // These sessions add noise/volume for realistic retrieval evaluation.
    Session {
        id: "misc_10",
        turns: &[
            Turn {
                role: "user",
                content: "I'm thinking about adopting another dog, maybe a border collie.",
            },
            Turn {
                role: "assistant",
                content: "Border collies are very intelligent but high-energy.",
            },
            Turn {
                role: "user",
                content: "Biscuit our golden retriever might enjoy the company.",
            },
            Turn {
                role: "assistant",
                content: "A companion dog can be great.",
            },
            Turn {
                role: "user",
                content: "Emma thinks we should wait until Sofia is older.",
            },
            Turn {
                role: "assistant",
                content: "That's sensible timing.",
            },
            Turn {
                role: "user",
                content: "I've been looking at dog breeders in the Pacific Northwest.",
            },
            Turn {
                role: "assistant",
                content: "There are many reputable breeders in the region.",
            },
            Turn {
                role: "user",
                content: "Alternatively we might adopt from the Oregon Humane Society.",
            },
            Turn {
                role: "assistant",
                content: "Adoption is a wonderful option.",
            },
            Turn {
                role: "user",
                content: "The dog would need a fenced yard, which we have.",
            },
            Turn {
                role: "assistant",
                content: "Good, that's an important requirement.",
            },
            Turn {
                role: "user",
                content: "I like the idea of a smaller second dog to balance energy levels.",
            },
            Turn {
                role: "assistant",
                content: "Size compatibility is a real consideration.",
            },
            Turn {
                role: "user",
                content: "Maybe a miniature schnauzer or a beagle.",
            },
            Turn {
                role: "assistant",
                content: "Both are great choices.",
            },
            Turn {
                role: "user",
                content: "We'll decide after the Portland Marathon in October.",
            },
            Turn {
                role: "assistant",
                content: "Good to have a timeline.",
            },
            Turn {
                role: "user",
                content: "Sofia has been asking for a cat but we're a dog family.",
            },
            Turn {
                role: "assistant",
                content: "A clear household preference.",
            },
        ],
    },
    Session {
        id: "misc_11",
        turns: &[
            Turn {
                role: "user",
                content: "I'm refinancing my mortgage to take advantage of current rates.",
            },
            Turn {
                role: "assistant",
                content: "Refinancing can save substantial money.",
            },
            Turn {
                role: "user",
                content: "Our home in Portland was purchased for 750000 in 2018.",
            },
            Turn {
                role: "assistant",
                content: "Portland real estate has moved significantly since then.",
            },
            Turn {
                role: "user",
                content: "The current estimated value is around 950000.",
            },
            Turn {
                role: "assistant",
                content: "A good appreciation in value.",
            },
            Turn {
                role: "user",
                content: "We're working with First Tech Federal Credit Union for the refi.",
            },
            Turn {
                role: "assistant",
                content: "First Tech is well-regarded.",
            },
            Turn {
                role: "user",
                content: "The new rate would be 6.2 percent 30-year fixed.",
            },
            Turn {
                role: "assistant",
                content: "That's a substantial improvement if your current rate is higher.",
            },
            Turn {
                role: "user",
                content: "Our original rate was 4.1 percent back in 2018.",
            },
            Turn {
                role: "assistant",
                content: "Ah, your current rate is actually lower, so refinancing may not make sense.",
            },
            Turn {
                role: "user",
                content: "You're right — we decided not to proceed with the refinance.",
            },
            Turn {
                role: "assistant",
                content: "Smart decision given the rate environment.",
            },
            Turn {
                role: "user",
                content: "Instead we're putting extra money toward the principal each month.",
            },
            Turn {
                role: "assistant",
                content: "Paying down principal is an excellent strategy.",
            },
            Turn {
                role: "user",
                content: "We have about 18 years left on the mortgage.",
            },
            Turn {
                role: "assistant",
                content: "Extra principal payments can shave years off that.",
            },
            Turn {
                role: "user",
                content: "Our financial advisor is Sandra Kim at Vanguard.",
            },
            Turn {
                role: "assistant",
                content: "Vanguard is known for low-cost index fund investing.",
            },
        ],
    },
    Session {
        id: "misc_12",
        turns: &[
            Turn {
                role: "user",
                content: "I'm learning Spanish with Duolingo — 6 months in now.",
            },
            Turn {
                role: "assistant",
                content: "Six months is a good foundation.",
            },
            Turn {
                role: "user",
                content: "I'm at about A2 level, can hold basic conversations.",
            },
            Turn {
                role: "assistant",
                content: "A2 is functional for travel.",
            },
            Turn {
                role: "user",
                content: "I want to visit Mexico City next year and use my Spanish there.",
            },
            Turn {
                role: "assistant",
                content: "Mexico City is a fantastic destination.",
            },
            Turn {
                role: "user",
                content: "Emma is also learning, we practice together over dinner.",
            },
            Turn {
                role: "assistant",
                content: "Learning together is motivating.",
            },
            Turn {
                role: "user",
                content: "My Duolingo streak is currently 183 days.",
            },
            Turn {
                role: "assistant",
                content: "Impressive consistency.",
            },
            Turn {
                role: "user",
                content: "I supplement with a Spanish podcast called Dreaming Spanish.",
            },
            Turn {
                role: "assistant",
                content: "Comprehensible input methods work well.",
            },
            Turn {
                role: "user",
                content: "My goal is B1 level by the end of next year.",
            },
            Turn {
                role: "assistant",
                content: "An achievable goal with your current pace.",
            },
            Turn {
                role: "user",
                content: "I also know a bit of Japanese from my Tokyo trip prep.",
            },
            Turn {
                role: "assistant",
                content: "Basic Japanese goes a long way in Tokyo.",
            },
            Turn {
                role: "user",
                content: "I can read hiragana and katakana but not kanji.",
            },
            Turn {
                role: "assistant",
                content: "Hiragana and katakana are the essential scripts.",
            },
            Turn {
                role: "user",
                content: "Sofia is picking up some Spanish words from us.",
            },
            Turn {
                role: "assistant",
                content: "Children absorb language effortlessly.",
            },
        ],
    },
    Session {
        id: "misc_13",
        turns: &[
            Turn {
                role: "user",
                content: "I've been having lower back pain for the past few months.",
            },
            Turn {
                role: "assistant",
                content: "Lower back pain is very common, especially for desk workers.",
            },
            Turn {
                role: "user",
                content: "My physiotherapist Dr. James Wu recommended specific exercises.",
            },
            Turn {
                role: "assistant",
                content: "Physical therapy is often more effective than medication.",
            },
            Turn {
                role: "user",
                content: "The exercises include bird-dog, dead bug, and cat-cow stretches.",
            },
            Turn {
                role: "assistant",
                content: "Classic core stabilization exercises.",
            },
            Turn {
                role: "user",
                content: "I upgraded to a standing desk at the office — a Uplift V2.",
            },
            Turn {
                role: "assistant",
                content: "Uplift desks are well-reviewed.",
            },
            Turn {
                role: "user",
                content: "I alternate sitting and standing every 30 minutes.",
            },
            Turn {
                role: "assistant",
                content: "Frequent position changes are important.",
            },
            Turn {
                role: "user",
                content: "My pain has reduced significantly over three weeks.",
            },
            Turn {
                role: "assistant",
                content: "Great progress.",
            },
            Turn {
                role: "user",
                content: "Dr. Wu also recommended swimming for low-impact cardio.",
            },
            Turn {
                role: "assistant",
                content: "Swimming is excellent for back health.",
            },
            Turn {
                role: "user",
                content: "I joined the YMCA downtown and go swimming on Saturday mornings.",
            },
            Turn {
                role: "assistant",
                content: "Consistent swimming makes a real difference.",
            },
            Turn {
                role: "user",
                content: "I'm being cautious with running until the back pain fully resolves.",
            },
            Turn {
                role: "assistant",
                content: "Smart to prioritize recovery.",
            },
            Turn {
                role: "user",
                content: "The Portland Marathon training is slightly behind schedule because of this.",
            },
            Turn {
                role: "assistant",
                content: "Better to arrive healthy than to risk injury.",
            },
        ],
    },
    Session {
        id: "misc_14",
        turns: &[
            Turn {
                role: "user",
                content: "Sofia started kindergarten at Buckman Elementary this fall.",
            },
            Turn {
                role: "assistant",
                content: "A big milestone!",
            },
            Turn {
                role: "user",
                content: "She loves her teacher Ms. Rodriguez and talks about school all the time.",
            },
            Turn {
                role: "assistant",
                content: "A great teacher makes all the difference.",
            },
            Turn {
                role: "user",
                content: "Buckman is only a 10-minute walk from our house.",
            },
            Turn {
                role: "assistant",
                content: "Convenient for the school commute.",
            },
            Turn {
                role: "user",
                content: "Sofia made two best friends: Anika and Oliver.",
            },
            Turn {
                role: "assistant",
                content: "Making friends is such an important part of kindergarten.",
            },
            Turn {
                role: "user",
                content: "She's already reading simple books — three-letter words and sight words.",
            },
            Turn {
                role: "assistant",
                content: "Early readers often have a lifelong love of books.",
            },
            Turn {
                role: "user",
                content: "I read to her every night before bed — currently reading Charlotte's Web.",
            },
            Turn {
                role: "assistant",
                content: "A classic. She'll love it.",
            },
            Turn {
                role: "user",
                content: "We signed her up for a Saturday art class at the Portland Art Museum.",
            },
            Turn {
                role: "assistant",
                content: "Art education is wonderful for creativity.",
            },
            Turn {
                role: "user",
                content: "She also expressed interest in ballet but we haven't decided yet.",
            },
            Turn {
                role: "assistant",
                content: "There's no rush — let her explore.",
            },
            Turn {
                role: "user",
                content: "Emma handles the morning school drop-off most days.",
            },
            Turn {
                role: "assistant",
                content: "A good division of parenting duties.",
            },
            Turn {
                role: "user",
                content: "I do pickup on Tuesdays and Thursdays.",
            },
            Turn {
                role: "assistant",
                content: "Consistent pickup days help kids feel secure.",
            },
        ],
    },
    Session {
        id: "misc_15",
        turns: &[
            Turn {
                role: "user",
                content: "I'm considering switching from VS Code to Neovim for development.",
            },
            Turn {
                role: "assistant",
                content: "Neovim has a steep learning curve but high ceiling.",
            },
            Turn {
                role: "user",
                content: "I've been using VS Code for five years and know it well.",
            },
            Turn {
                role: "assistant",
                content: "Switching editors is a big productivity investment.",
            },
            Turn {
                role: "user",
                content: "Several colleagues at QuantumLogic swear by Neovim.",
            },
            Turn {
                role: "assistant",
                content: "Peer influence is a real factor in tool adoption.",
            },
            Turn {
                role: "user",
                content: "I tried it for a week and the modal editing is natural for me now.",
            },
            Turn {
                role: "assistant",
                content: "Once it clicks, modal editing is very efficient.",
            },
            Turn {
                role: "user",
                content: "I use LazyVim as my configuration framework.",
            },
            Turn {
                role: "assistant",
                content: "LazyVim is a great starting point.",
            },
            Turn {
                role: "user",
                content: "The Rust LSP integration with rust-analyzer works perfectly in Neovim.",
            },
            Turn {
                role: "assistant",
                content: "rust-analyzer in Neovim is excellent.",
            },
            Turn {
                role: "user",
                content: "I'm keeping VS Code around for pair programming sessions.",
            },
            Turn {
                role: "assistant",
                content: "Live Share is a compelling VS Code feature.",
            },
            Turn {
                role: "user",
                content: "I also use Helix sometimes for quick edits in the terminal.",
            },
            Turn {
                role: "assistant",
                content: "Helix has a clean design.",
            },
            Turn {
                role: "user",
                content: "My terminal is Alacritty with tmux for session management.",
            },
            Turn {
                role: "assistant",
                content: "A classic terminal setup.",
            },
            Turn {
                role: "user",
                content: "My dotfiles are public on GitHub at github.com/alex-mitchell-pdx.",
            },
            Turn {
                role: "assistant",
                content: "I'll note that for future reference.",
            },
        ],
    },
    Session {
        id: "misc_16",
        turns: &[
            Turn {
                role: "user",
                content: "We renovated our kitchen last spring — cost about 45000 dollars.",
            },
            Turn {
                role: "assistant",
                content: "Kitchen renovations can be a great investment.",
            },
            Turn {
                role: "user",
                content: "We installed quartz countertops in Calacatta Gold pattern.",
            },
            Turn {
                role: "assistant",
                content: "Calacatta quartz is a beautiful choice.",
            },
            Turn {
                role: "user",
                content: "The cabinets are custom shaker style in a sage green color.",
            },
            Turn {
                role: "assistant",
                content: "Sage green kitchens are very popular right now.",
            },
            Turn {
                role: "user",
                content: "We hired a local contractor named Marcus Powell — excellent work.",
            },
            Turn {
                role: "assistant",
                content: "A good contractor makes all the difference.",
            },
            Turn {
                role: "user",
                content: "The renovation took six weeks and we ate takeout the whole time.",
            },
            Turn {
                role: "assistant",
                content: "A common trade-off during kitchen renos.",
            },
            Turn {
                role: "user",
                content: "We added an island with seating for three people.",
            },
            Turn {
                role: "assistant",
                content: "Kitchen islands are very functional.",
            },
            Turn {
                role: "user",
                content: "The appliances are all Bosch — refrigerator, dishwasher, and range.",
            },
            Turn {
                role: "assistant",
                content: "Bosch appliances are reliable and quiet.",
            },
            Turn {
                role: "user",
                content: "The hardwood floors throughout the kitchen are white oak.",
            },
            Turn {
                role: "assistant",
                content: "White oak is a classic durable choice.",
            },
            Turn {
                role: "user",
                content: "We're very happy with the result — it increased the home's value.",
            },
            Turn {
                role: "assistant",
                content: "Kitchen renovations typically provide good ROI.",
            },
            Turn {
                role: "user",
                content: "Next project will be updating the master bathroom in a few years.",
            },
            Turn {
                role: "assistant",
                content: "Bathrooms are also high-value renovation projects.",
            },
        ],
    },
    Session {
        id: "misc_17",
        turns: &[
            Turn {
                role: "user",
                content: "I invested in some index funds recently — Vanguard VTSAX and VTI.",
            },
            Turn {
                role: "assistant",
                content: "Both are excellent total market index funds.",
            },
            Turn {
                role: "user",
                content: "I put 20 percent of my paycheck into my 401k at QuantumLogic.",
            },
            Turn {
                role: "assistant",
                content: "Maxing out tax-advantaged accounts is smart.",
            },
            Turn {
                role: "user",
                content: "The company matches up to 6 percent which I always capture.",
            },
            Turn {
                role: "assistant",
                content: "Never leave free money on the table.",
            },
            Turn {
                role: "user",
                content: "I also have a Roth IRA maxed out each year.",
            },
            Turn {
                role: "assistant",
                content: "Roth IRA is particularly valuable for high earners.",
            },
            Turn {
                role: "user",
                content: "My financial goal is to retire at 55 if possible.",
            },
            Turn {
                role: "assistant",
                content: "An achievable target with disciplined savings at your income level.",
            },
            Turn {
                role: "user",
                content: "I'm interested in real estate as a second income stream eventually.",
            },
            Turn {
                role: "assistant",
                content: "Rental properties can provide good cash flow.",
            },
            Turn {
                role: "user",
                content: "Sandra Kim at Vanguard says I'm on track for retirement at 55.",
            },
            Turn {
                role: "assistant",
                content: "Good to have professional validation of your plan.",
            },
            Turn {
                role: "user",
                content: "I keep six months of expenses in a high-yield savings account as emergency fund.",
            },
            Turn {
                role: "assistant",
                content: "Six months is the standard recommendation.",
            },
            Turn {
                role: "user",
                content: "My investment portfolio is currently 90 percent stocks and 10 percent bonds.",
            },
            Turn {
                role: "assistant",
                content: "Appropriate allocation for a 34-year-old with long horizon.",
            },
            Turn {
                role: "user",
                content: "I review my portfolio quarterly with Sandra.",
            },
            Turn {
                role: "assistant",
                content: "Quarterly reviews are a good cadence.",
            },
        ],
    },
    Session {
        id: "misc_18",
        turns: &[
            Turn {
                role: "user",
                content: "I'm taking a two-week sabbatical in December to recharge.",
            },
            Turn {
                role: "assistant",
                content: "Sabbaticals are important for sustainable high performance.",
            },
            Turn {
                role: "user",
                content: "I plan to stay off email and Slack the entire time.",
            },
            Turn {
                role: "assistant",
                content: "Full disconnection is the only way to truly recharge.",
            },
            Turn {
                role: "user",
                content: "We're thinking about flying to Costa Rica for a family vacation during that time.",
            },
            Turn {
                role: "assistant",
                content: "Costa Rica is great for families.",
            },
            Turn {
                role: "user",
                content: "Sofia would love the wildlife — she's obsessed with monkeys.",
            },
            Turn {
                role: "assistant",
                content: "Costa Rica has howler monkeys everywhere.",
            },
            Turn {
                role: "user",
                content: "We'd stay in the Arenal volcano area and also near Manuel Antonio beach.",
            },
            Turn {
                role: "assistant",
                content: "Two great contrasting environments.",
            },
            Turn {
                role: "user",
                content: "Emma is handling all the planning for the Costa Rica trip.",
            },
            Turn {
                role: "assistant",
                content: "Nice when partners divide the planning work.",
            },
            Turn {
                role: "user",
                content: "We'd leave December 20th and return January 3rd.",
            },
            Turn {
                role: "assistant",
                content: "A lovely holiday stretch.",
            },
            Turn {
                role: "user",
                content: "My mother Margaret will dog-sit Biscuit while we're away.",
            },
            Turn {
                role: "assistant",
                content: "Margaret to the rescue again.",
            },
            Turn {
                role: "user",
                content: "I've never been to Central America before, very excited.",
            },
            Turn {
                role: "assistant",
                content: "It will be a memorable first visit.",
            },
            Turn {
                role: "user",
                content: "I'm hoping to do a canopy zipline tour in Monteverde.",
            },
            Turn {
                role: "assistant",
                content: "Monteverde is the ultimate zipline destination.",
            },
        ],
    },
    Session {
        id: "misc_19",
        turns: &[
            Turn {
                role: "user",
                content: "My side project is building an open source Rust crate for distributed tracing.",
            },
            Turn {
                role: "assistant",
                content: "Distributed tracing is a valuable tool.",
            },
            Turn {
                role: "user",
                content: "The crate is called tracewing and it's compatible with OpenTelemetry.",
            },
            Turn {
                role: "assistant",
                content: "OpenTelemetry compatibility is important for adoption.",
            },
            Turn {
                role: "user",
                content: "I work on tracewing on Sunday mornings for about two hours.",
            },
            Turn {
                role: "assistant",
                content: "Dedicated side-project time is how things get done.",
            },
            Turn {
                role: "user",
                content: "The crate already has 300 stars on GitHub.",
            },
            Turn {
                role: "assistant",
                content: "That's solid early traction.",
            },
            Turn {
                role: "user",
                content: "My collaborator on tracewing is Wei Zhang from the Rust foundation community.",
            },
            Turn {
                role: "assistant",
                content: "Open source collaboration is powerful.",
            },
            Turn {
                role: "user",
                content: "We're planning tracewing version 0.3 with async span support.",
            },
            Turn {
                role: "assistant",
                content: "Async support will significantly expand the use cases.",
            },
            Turn {
                role: "user",
                content: "I use tracewing internally at QuantumLogic and it saves us hours of debugging.",
            },
            Turn {
                role: "assistant",
                content: "Dogfooding your own tool validates it.",
            },
            Turn {
                role: "user",
                content: "I presented tracewing at the Portland Rust meetup last month.",
            },
            Turn {
                role: "assistant",
                content: "Meetup presentations are great for visibility.",
            },
            Turn {
                role: "user",
                content: "Dr. Huang wants to officially adopt tracewing as a QuantumLogic standard library.",
            },
            Turn {
                role: "assistant",
                content: "Official adoption is significant.",
            },
            Turn {
                role: "user",
                content: "I might formalize it with a proper GitHub organization called tracewing-rs.",
            },
            Turn {
                role: "assistant",
                content: "A dedicated org adds credibility.",
            },
        ],
    },
    // Filler sessions 20-49 (brief, adding corpus volume)
    Session {
        id: "filler_20",
        turns: &[
            Turn {
                role: "user",
                content: "I upgraded to an iPhone 16 Pro this year.",
            },
            Turn {
                role: "assistant",
                content: "The iPhone 16 Pro has an excellent camera.",
            },
            Turn {
                role: "user",
                content: "I primarily use it for photos of Sofia and our travel.",
            },
            Turn {
                role: "assistant",
                content: "A great use case.",
            },
            Turn {
                role: "user",
                content: "I also use a Sony A7 IV for serious photography.",
            },
            Turn {
                role: "assistant",
                content: "The A7 IV is a professional-grade camera.",
            },
            Turn {
                role: "user",
                content: "I got into photography during the Tokyo trip.",
            },
            Turn {
                role: "assistant",
                content: "Tokyo is a photographer's paradise.",
            },
            Turn {
                role: "user",
                content: "My favorite lens is the Sony 24-70mm f/2.8 GM.",
            },
            Turn {
                role: "assistant",
                content: "The 24-70 GM is a versatile workhorse.",
            },
            Turn {
                role: "user",
                content: "I post photos on Instagram at @alexmitchell.pdx.",
            },
            Turn {
                role: "assistant",
                content: "I'll note your Instagram handle.",
            },
            Turn {
                role: "user",
                content: "I edit photos in Lightroom on my MacBook Pro.",
            },
            Turn {
                role: "assistant",
                content: "Lightroom is the industry standard.",
            },
            Turn {
                role: "user",
                content: "My MacBook Pro is the M3 Max 16-inch model.",
            },
            Turn {
                role: "assistant",
                content: "Excellent machine for photo and development work.",
            },
            Turn {
                role: "user",
                content: "I also have a Mac Mini M2 Pro as a home server.",
            },
            Turn {
                role: "assistant",
                content: "Mac Mini makes a great low-power server.",
            },
            Turn {
                role: "user",
                content: "I self-host a Nextcloud instance for family photos.",
            },
            Turn {
                role: "assistant",
                content: "Self-hosting Nextcloud is a privacy-conscious choice.",
            },
        ],
    },
    Session {
        id: "filler_21",
        turns: &[
            Turn {
                role: "user",
                content: "I just finished reading a fascinating book about distributed systems.",
            },
            Turn {
                role: "assistant",
                content: "Which book?",
            },
            Turn {
                role: "user",
                content: "Designing Data-Intensive Applications by Martin Kleppmann.",
            },
            Turn {
                role: "assistant",
                content: "A classic in the field.",
            },
            Turn {
                role: "user",
                content: "I highlighted almost every page — it's incredibly dense with knowledge.",
            },
            Turn {
                role: "assistant",
                content: "Kleppmann writes with remarkable clarity.",
            },
            Turn {
                role: "user",
                content: "I'm now reading A Philosophy of Software Design by John Ousterhout.",
            },
            Turn {
                role: "assistant",
                content: "Ousterhout has strong opinions on software design.",
            },
            Turn {
                role: "user",
                content: "I track my reading on Goodreads — profile alexmitchell_reads.",
            },
            Turn {
                role: "assistant",
                content: "Goodreads is a good way to maintain reading lists.",
            },
            Turn {
                role: "user",
                content: "I'm in a technical book club with colleagues at QuantumLogic.",
            },
            Turn {
                role: "assistant",
                content: "Book clubs keep you accountable to finish books.",
            },
            Turn {
                role: "user",
                content: "We meet on the last Friday of each month for book club.",
            },
            Turn {
                role: "assistant",
                content: "A consistent cadence.",
            },
            Turn {
                role: "user",
                content: "The next book club selection is The Pragmatic Programmer.",
            },
            Turn {
                role: "assistant",
                content: "Another timeless classic.",
            },
            Turn {
                role: "user",
                content: "I donate books I've finished to the Multnomah County Library.",
            },
            Turn {
                role: "assistant",
                content: "A generous contribution to the community.",
            },
            Turn {
                role: "user",
                content: "I have a physical library of about 300 technical books at home.",
            },
            Turn {
                role: "assistant",
                content: "An impressive collection.",
            },
        ],
    },
    Session {
        id: "filler_22",
        turns: &[
            Turn {
                role: "user",
                content: "Portland weather has been unusually warm this October.",
            },
            Turn {
                role: "assistant",
                content: "Unusual October warmth seems to be more common.",
            },
            Turn {
                role: "user",
                content: "We've had several 75-degree days which is very rare.",
            },
            Turn {
                role: "assistant",
                content: "Enjoy it while it lasts.",
            },
            Turn {
                role: "user",
                content: "Normally October in Portland means rain starting mid-month.",
            },
            Turn {
                role: "assistant",
                content: "The rainy season usually begins in earnest by Halloween.",
            },
            Turn {
                role: "user",
                content: "I love Portland in the fall when the leaves turn on the park blocks.",
            },
            Turn {
                role: "assistant",
                content: "Portland's tree-lined streets are beautiful in fall colors.",
            },
            Turn {
                role: "user",
                content: "Biscuit gets extra muddy on walks when the rain comes.",
            },
            Turn {
                role: "assistant",
                content: "Golden retrievers plus mud is a classic combination.",
            },
            Turn {
                role: "user",
                content: "We have a dedicated dog towel station by the back door.",
            },
            Turn {
                role: "assistant",
                content: "A practical Portland-dog-owner solution.",
            },
            Turn {
                role: "user",
                content: "I'm looking forward to the first snow of the season.",
            },
            Turn {
                role: "assistant",
                content: "Portland doesn't get much snow but it's magical when it does.",
            },
            Turn {
                role: "user",
                content: "Last winter we got three inches in January and Sofia built her first snowman.",
            },
            Turn {
                role: "assistant",
                content: "A memorable first snowman moment.",
            },
            Turn {
                role: "user",
                content: "We live at a higher elevation in the NE Portland hills so we get more snow.",
            },
            Turn {
                role: "assistant",
                content: "Elevation makes a real difference for Portland snow.",
            },
            Turn {
                role: "user",
                content: "Our neighborhood association coordinates snow clearing on private roads.",
            },
            Turn {
                role: "assistant",
                content: "Community coordination is important for those conditions.",
            },
        ],
    },
    Session {
        id: "filler_23",
        turns: &[
            Turn {
                role: "user",
                content: "I've been experimenting with home automation using Home Assistant.",
            },
            Turn {
                role: "assistant",
                content: "Home Assistant is powerful for local control.",
            },
            Turn {
                role: "user",
                content: "I have smart thermostats, door locks, and light switches throughout the house.",
            },
            Turn {
                role: "assistant",
                content: "Good coverage for a smart home.",
            },
            Turn {
                role: "user",
                content: "All devices run locally with no cloud dependency.",
            },
            Turn {
                role: "assistant",
                content: "Local-only is the privacy-conscious choice.",
            },
            Turn {
                role: "user",
                content: "I use Zigbee for the lights and Z-Wave for the door locks.",
            },
            Turn {
                role: "assistant",
                content: "Good protocol choices for reliability.",
            },
            Turn {
                role: "user",
                content: "The whole system runs on my Mac Mini M2 Pro home server.",
            },
            Turn {
                role: "assistant",
                content: "A capable and quiet server for Home Assistant.",
            },
            Turn {
                role: "user",
                content: "I have an automation that dims the lights at Sofia's bedtime.",
            },
            Turn {
                role: "assistant",
                content: "Smart bedtime routines are helpful for kids.",
            },
            Turn {
                role: "user",
                content: "The door locks automatically at 11pm every night.",
            },
            Turn {
                role: "assistant",
                content: "Good security habit.",
            },
            Turn {
                role: "user",
                content: "I'm considering adding cameras but Emma is concerned about privacy.",
            },
            Turn {
                role: "assistant",
                content: "Indoor cameras are a common marital negotiation topic.",
            },
            Turn {
                role: "user",
                content: "We compromised on a doorbell camera only — a Reolink.",
            },
            Turn {
                role: "assistant",
                content: "A reasonable middle ground.",
            },
            Turn {
                role: "user",
                content: "I documented the whole setup on my personal blog.",
            },
            Turn {
                role: "assistant",
                content: "Sharing home automation setups helps the community.",
            },
        ],
    },
    Session {
        id: "filler_24",
        turns: &[
            Turn {
                role: "user",
                content: "I had coffee this morning with my old MIT classmate David Park.",
            },
            Turn {
                role: "assistant",
                content: "Nice to reconnect with old classmates.",
            },
            Turn {
                role: "user",
                content: "David is now CTO at a Series B startup in Seattle called Neurovex.",
            },
            Turn {
                role: "assistant",
                content: "CTO of a Series B is a significant role.",
            },
            Turn {
                role: "user",
                content: "He's trying to recruit me to join as VP of Engineering.",
            },
            Turn {
                role: "assistant",
                content: "That's a flattering offer.",
            },
            Turn {
                role: "user",
                content: "The role would come with a significant equity package.",
            },
            Turn {
                role: "assistant",
                content: "Equity can be very valuable at that stage.",
            },
            Turn {
                role: "user",
                content: "But I'd have to relocate to Seattle which is complicated with Sofia in kindergarten.",
            },
            Turn {
                role: "assistant",
                content: "School transitions are a real constraint for families.",
            },
            Turn {
                role: "user",
                content: "I'm going to think about it but I'm leaning toward staying at QuantumLogic.",
            },
            Turn {
                role: "assistant",
                content: "Stability has real value too.",
            },
            Turn {
                role: "user",
                content: "The timing aligns with my promotion review in November.",
            },
            Turn {
                role: "assistant",
                content: "You could use the competing offer as leverage.",
            },
            Turn {
                role: "user",
                content: "I'd need to decide before Thanksgiving — David needs an answer.",
            },
            Turn {
                role: "assistant",
                content: "A firm deadline helps clarify priorities.",
            },
            Turn {
                role: "user",
                content: "Emma is supportive either way, which I appreciate.",
            },
            Turn {
                role: "assistant",
                content: "A supportive partner makes these decisions easier.",
            },
            Turn {
                role: "user",
                content: "I ultimately declined the Neurovex offer to focus on the Prometheus project.",
            },
            Turn {
                role: "assistant",
                content: "A clear-headed decision.",
            },
        ],
    },
    // Sessions 25-49: additional noise topics
    Session {
        id: "filler_25",
        turns: &[
            Turn {
                role: "user",
                content: "I maintain a personal wiki in Obsidian for knowledge management.",
            },
            Turn {
                role: "assistant",
                content: "Obsidian is excellent for connected note-taking.",
            },
            Turn {
                role: "user",
                content: "I have over 2000 notes accumulated over three years.",
            },
            Turn {
                role: "assistant",
                content: "A substantial knowledge base.",
            },
            Turn {
                role: "user",
                content: "My main vault is synced via iCloud across my devices.",
            },
            Turn {
                role: "assistant",
                content: "iCloud sync is seamless for Apple ecosystem users.",
            },
            Turn {
                role: "user",
                content: "I use the Zettelkasten method for linking ideas.",
            },
            Turn {
                role: "assistant",
                content: "Zettelkasten creates emergent insights through linking.",
            },
            Turn {
                role: "user",
                content: "I write a weekly review note every Sunday evening.",
            },
            Turn {
                role: "assistant",
                content: "Weekly reviews are a cornerstone of effective knowledge work.",
            },
            Turn {
                role: "user",
                content: "I also keep meeting notes there with action items tagged.",
            },
            Turn {
                role: "assistant",
                content: "Structured meeting notes with action items are essential.",
            },
            Turn {
                role: "user",
                content: "I shared my Obsidian setup template on Twitter and it got 10k likes.",
            },
            Turn {
                role: "assistant",
                content: "Productivity content resonates widely.",
            },
            Turn {
                role: "user",
                content: "I'm working on a blog post series about my note-taking system.",
            },
            Turn {
                role: "assistant",
                content: "People are always looking for better PKM systems.",
            },
            Turn {
                role: "user",
                content: "My blog is at alexmitchell.dev and has about 5000 monthly readers.",
            },
            Turn {
                role: "assistant",
                content: "A solid readership for a technical blog.",
            },
            Turn {
                role: "user",
                content: "I write one technical post per month on average.",
            },
            Turn {
                role: "assistant",
                content: "Consistent output builds audience.",
            },
        ],
    },
    Session {
        id: "filler_26",
        turns: &[
            Turn {
                role: "user",
                content: "I recently completed a 72-hour water fast for health reasons.",
            },
            Turn {
                role: "assistant",
                content: "Extended fasting should be done carefully.",
            },
            Turn {
                role: "user",
                content: "I worked with my doctor Dr. Chen at OHSU before attempting it.",
            },
            Turn {
                role: "assistant",
                content: "Medical supervision is important for extended fasts.",
            },
            Turn {
                role: "user",
                content: "The benefits were improved mental clarity and reset appetite.",
            },
            Turn {
                role: "assistant",
                content: "Those are commonly reported benefits.",
            },
            Turn {
                role: "user",
                content: "I do intermittent fasting daily — 16:8 protocol.",
            },
            Turn {
                role: "assistant",
                content: "16:8 is one of the most sustainable fasting protocols.",
            },
            Turn {
                role: "user",
                content: "My eating window is noon to 8pm every day.",
            },
            Turn {
                role: "assistant",
                content: "That aligns well with typical work and family schedules.",
            },
            Turn {
                role: "user",
                content: "I track my health metrics with a Whoop 4.0 band.",
            },
            Turn {
                role: "assistant",
                content: "Whoop provides detailed recovery and strain data.",
            },
            Turn {
                role: "user",
                content: "My resting heart rate averages 48 bpm which is athletic.",
            },
            Turn {
                role: "assistant",
                content: "That's excellent cardiovascular fitness.",
            },
            Turn {
                role: "user",
                content: "I get about 7.5 hours of sleep on average.",
            },
            Turn {
                role: "assistant",
                content: "7.5 hours is solid sleep for a high performer.",
            },
            Turn {
                role: "user",
                content: "My HRV average is 68 which Whoop rates as green zone.",
            },
            Turn {
                role: "assistant",
                content: "HRV in the green zone is a good sign of recovery.",
            },
            Turn {
                role: "user",
                content: "I avoid alcohol almost entirely — maybe two drinks per month.",
            },
            Turn {
                role: "assistant",
                content: "Low alcohol intake has clear health benefits.",
            },
        ],
    },
    Session {
        id: "filler_27",
        turns: &[
            Turn {
                role: "user",
                content: "I coach Sofia's Saturday morning soccer team.",
            },
            Turn {
                role: "assistant",
                content: "Youth soccer coaching is rewarding.",
            },
            Turn {
                role: "user",
                content: "The team is called the Purple Dragons — 5-year-olds.",
            },
            Turn {
                role: "assistant",
                content: "The Purple Dragons sounds adorable.",
            },
            Turn {
                role: "user",
                content: "We focus entirely on fun and basic skills — no competition at this age.",
            },
            Turn {
                role: "assistant",
                content: "The right approach for 5-year-olds.",
            },
            Turn {
                role: "user",
                content: "Sofia is one of the more coordinated players on the team.",
            },
            Turn {
                role: "assistant",
                content: "Her running background probably helps.",
            },
            Turn {
                role: "user",
                content: "We have 12 kids on the roster and three parent volunteers.",
            },
            Turn {
                role: "assistant",
                content: "Good volunteer ratio.",
            },
            Turn {
                role: "user",
                content: "We practice at Kenilworth Park near our house.",
            },
            Turn {
                role: "assistant",
                content: "A convenient local field.",
            },
            Turn {
                role: "user",
                content: "Emma brings orange slices and juice boxes for after practice.",
            },
            Turn {
                role: "assistant",
                content: "The classic youth soccer snack tradition.",
            },
            Turn {
                role: "user",
                content: "I played soccer through high school back in Boston.",
            },
            Turn {
                role: "assistant",
                content: "Your background helps in coaching.",
            },
            Turn {
                role: "user",
                content: "The season runs from September through November.",
            },
            Turn {
                role: "assistant",
                content: "A manageable autumn commitment.",
            },
            Turn {
                role: "user",
                content: "We have a year-end pizza party at Flying Pie Pizzaria.",
            },
            Turn {
                role: "assistant",
                content: "A great way to end the season.",
            },
        ],
    },
    Session {
        id: "filler_28",
        turns: &[
            Turn {
                role: "user",
                content: "I bought a new electric car last month — a Rivian R1T pickup.",
            },
            Turn {
                role: "assistant",
                content: "The Rivian R1T is a fascinating EV.",
            },
            Turn {
                role: "user",
                content: "We traded in Emma's old Subaru Outback for it.",
            },
            Turn {
                role: "assistant",
                content: "A significant upgrade.",
            },
            Turn {
                role: "user",
                content: "The range is about 320 miles which works perfectly for our lifestyle.",
            },
            Turn {
                role: "assistant",
                content: "320 miles is very practical range.",
            },
            Turn {
                role: "user",
                content: "We installed a Level 2 charger in our garage — ChargePoint Home Flex.",
            },
            Turn {
                role: "assistant",
                content: "Home charging is very convenient.",
            },
            Turn {
                role: "user",
                content: "The monthly energy bill went up about 40 dollars for the car charging.",
            },
            Turn {
                role: "assistant",
                content: "Much cheaper than equivalent gas costs.",
            },
            Turn {
                role: "user",
                content: "I use it for the morning commute and the drive to Seattle to see my mother.",
            },
            Turn {
                role: "assistant",
                content: "320 miles gets you to Seattle and back with a charge in between.",
            },
            Turn {
                role: "user",
                content: "Sofia calls it the camping car because of all the off-road features.",
            },
            Turn {
                role: "assistant",
                content: "The Rivian does have impressive overlanding capability.",
            },
            Turn {
                role: "user",
                content: "We're planning a camping trip to Mount Hood with the Rivian next summer.",
            },
            Turn {
                role: "assistant",
                content: "Mount Hood is a great destination.",
            },
            Turn {
                role: "user",
                content: "I also kept my old commuter bike for days when the weather is nice.",
            },
            Turn {
                role: "assistant",
                content: "A bike for nice days is very Portland.",
            },
            Turn {
                role: "user",
                content: "The Rivian came out to about 78000 dollars after federal EV credit.",
            },
            Turn {
                role: "assistant",
                content: "The federal credit made it more accessible.",
            },
        ],
    },
    Session {
        id: "filler_29",
        turns: &[
            Turn {
                role: "user",
                content: "I completed an online course on machine learning from fast.ai.",
            },
            Turn {
                role: "assistant",
                content: "fast.ai has excellent practical ML courses.",
            },
            Turn {
                role: "user",
                content: "The course is called Practical Deep Learning for Coders.",
            },
            Turn {
                role: "assistant",
                content: "Jeremy Howard's teaching style is very accessible.",
            },
            Turn {
                role: "user",
                content: "I implemented a basic image classifier for identifying birds.",
            },
            Turn {
                role: "assistant",
                content: "Image classification is a great first project.",
            },
            Turn {
                role: "user",
                content: "I used the bird dataset to classify Oregon native species.",
            },
            Turn {
                role: "assistant",
                content: "Localizing the dataset makes it more meaningful.",
            },
            Turn {
                role: "user",
                content: "The model achieved 94 percent accuracy on the test set.",
            },
            Turn {
                role: "assistant",
                content: "That's excellent for a first project.",
            },
            Turn {
                role: "user",
                content: "I'm interested in applying ML to the Prometheus project for anomaly detection.",
            },
            Turn {
                role: "assistant",
                content: "ML-based anomaly detection is valuable for streaming systems.",
            },
            Turn {
                role: "user",
                content: "I deployed the bird classifier as a Rust web service using tract-onnx.",
            },
            Turn {
                role: "assistant",
                content: "Rust inference with ONNX is very performant.",
            },
            Turn {
                role: "user",
                content: "Sofia uses the bird app on walks to identify birds — she loves it.",
            },
            Turn {
                role: "assistant",
                content: "Real-world use by a 5-year-old is the ultimate validation.",
            },
            Turn {
                role: "user",
                content: "I'm now taking the second part of the fast.ai course.",
            },
            Turn {
                role: "assistant",
                content: "The advanced course goes deep into the internals.",
            },
            Turn {
                role: "user",
                content: "I might write about the Rust deployment on my blog.",
            },
            Turn {
                role: "assistant",
                content: "That would be a unique and valuable post.",
            },
        ],
    },
    // sessions 30-49: brief repeated-topic noise sessions
    Session {
        id: "noise_30",
        turns: &[
            Turn {
                role: "user",
                content: "Reminder: team standup tomorrow at 9:15am.",
            },
            Turn {
                role: "assistant",
                content: "Noted, standup at 9:15am.",
            },
            Turn {
                role: "user",
                content: "Maria will present the Kafka integration progress.",
            },
            Turn {
                role: "assistant",
                content: "Looking forward to the update.",
            },
            Turn {
                role: "user",
                content: "Kenji has a demo of the Raft consensus layer ready.",
            },
            Turn {
                role: "assistant",
                content: "Exciting milestone.",
            },
            Turn {
                role: "user",
                content: "The sprint ends on Friday and we should hit our velocity target.",
            },
            Turn {
                role: "assistant",
                content: "Good sprint cadence.",
            },
            Turn {
                role: "user",
                content: "Need to block time for Prometheus code review Thursday afternoon.",
            },
            Turn {
                role: "assistant",
                content: "Code reviews are essential for quality.",
            },
            Turn {
                role: "user",
                content: "Dr. Huang wants a weekly status email every Monday by noon.",
            },
            Turn {
                role: "assistant",
                content: "A reasonable reporting rhythm.",
            },
            Turn {
                role: "user",
                content: "I set up a recurring calendar event for the status email.",
            },
            Turn {
                role: "assistant",
                content: "Good to systemize that.",
            },
            Turn {
                role: "user",
                content: "The Prometheus milestone demo is scheduled for October 25th.",
            },
            Turn {
                role: "assistant",
                content: "That's coming up quickly.",
            },
            Turn {
                role: "user",
                content: "We'll demo to Dr. Huang and the product stakeholders.",
            },
            Turn {
                role: "assistant",
                content: "A significant milestone review.",
            },
            Turn {
                role: "user",
                content: "I've prepared a slide deck covering architecture and performance benchmarks.",
            },
            Turn {
                role: "assistant",
                content: "Data-driven presentations work well for engineering reviews.",
            },
        ],
    },
    Session {
        id: "noise_31",
        turns: &[
            Turn {
                role: "user",
                content: "Sofia had a slight fever this morning so Emma stayed home with her.",
            },
            Turn {
                role: "assistant",
                content: "Hope Sofia feels better soon.",
            },
            Turn {
                role: "user",
                content: "The doctor said it's just a minor cold going around kindergarten.",
            },
            Turn {
                role: "assistant",
                content: "Kindergarten is a germ exchange hub.",
            },
            Turn {
                role: "user",
                content: "I made my morning run shorter today to help with the morning routine.",
            },
            Turn {
                role: "assistant",
                content: "Adapting to family needs.",
            },
            Turn {
                role: "user",
                content: "Emma made Sofia her favorite soup — chicken noodle.",
            },
            Turn {
                role: "assistant",
                content: "Classic remedy.",
            },
            Turn {
                role: "user",
                content: "They spent the afternoon watching Bluey on Disney Plus.",
            },
            Turn {
                role: "assistant",
                content: "Bluey is excellent television for kids.",
            },
            Turn {
                role: "user",
                content: "Sofia was back to normal by dinner, eating two full bowls of soup.",
            },
            Turn {
                role: "assistant",
                content: "Kids recover fast.",
            },
            Turn {
                role: "user",
                content: "She should be able to go back to school tomorrow.",
            },
            Turn {
                role: "assistant",
                content: "Good news.",
            },
            Turn {
                role: "user",
                content: "I rescheduled my afternoon meetings to work from home.",
            },
            Turn {
                role: "assistant",
                content: "Remote work flexibility is valuable for family situations.",
            },
            Turn {
                role: "user",
                content: "The back pain made working from the couch harder than expected.",
            },
            Turn {
                role: "assistant",
                content: "The standing desk is more ergonomic than the couch.",
            },
            Turn {
                role: "user",
                content: "I set up the laptop on the kitchen island with my portable monitor.",
            },
            Turn {
                role: "assistant",
                content: "Making the best of the situation.",
            },
        ],
    },
    Session {
        id: "noise_32",
        turns: &[
            Turn {
                role: "user",
                content: "The Portland Rust user group meetup is next Thursday evening.",
            },
            Turn {
                role: "assistant",
                content: "Are you presenting?",
            },
            Turn {
                role: "user",
                content: "Yes, I'm giving a 20-minute talk on memory-safe systems design.",
            },
            Turn {
                role: "assistant",
                content: "Memory safety is always a relevant Rust topic.",
            },
            Turn {
                role: "user",
                content: "The venue is Cloudflare's Portland office downtown.",
            },
            Turn {
                role: "assistant",
                content: "Cloudflare has been active in the Rust community.",
            },
            Turn {
                role: "user",
                content: "I'm expecting about 40-50 attendees.",
            },
            Turn {
                role: "assistant",
                content: "Good turnout for a local meetup.",
            },
            Turn {
                role: "user",
                content: "I'll talk about lessons learned from the Prometheus project.",
            },
            Turn {
                role: "assistant",
                content: "Practical lessons are highly valued.",
            },
            Turn {
                role: "user",
                content: "My slides are mostly code examples and performance charts.",
            },
            Turn {
                role: "assistant",
                content: "The Rust community appreciates concrete benchmarks.",
            },
            Turn {
                role: "user",
                content: "I'll mention tracewing during the Q&A if it comes up.",
            },
            Turn {
                role: "assistant",
                content: "Natural organic promotion.",
            },
            Turn {
                role: "user",
                content: "Wei Zhang from my tracewing project will also attend.",
            },
            Turn {
                role: "assistant",
                content: "Good to have a collaborator in the audience.",
            },
            Turn {
                role: "user",
                content: "There's a social hour at a nearby bar after the talks.",
            },
            Turn {
                role: "assistant",
                content: "Hallway conversations at meetups are often the most valuable.",
            },
            Turn {
                role: "user",
                content: "I'll bike to the venue since it's only 15 minutes from our house.",
            },
            Turn {
                role: "assistant",
                content: "Portland cycling culture.",
            },
        ],
    },
    Session {
        id: "noise_33",
        turns: &[
            Turn {
                role: "user",
                content: "I need to file my quarterly estimated taxes by September 15th.",
            },
            Turn {
                role: "assistant",
                content: "Q3 estimated taxes are due September 15th, correct.",
            },
            Turn {
                role: "user",
                content: "My accountant is James O'Connor at a local Portland CPA firm.",
            },
            Turn {
                role: "assistant",
                content: "A good accountant saves more than they cost.",
            },
            Turn {
                role: "user",
                content: "I have some freelance consulting income on top of my QuantumLogic salary.",
            },
            Turn {
                role: "assistant",
                content: "Freelance income requires careful tax planning.",
            },
            Turn {
                role: "user",
                content: "The consulting is about 30000 annually from two clients.",
            },
            Turn {
                role: "assistant",
                content: "A significant supplemental income.",
            },
            Turn {
                role: "user",
                content: "I use a sole proprietorship structure for the consulting work.",
            },
            Turn {
                role: "assistant",
                content: "Simple for your scale, though an S-Corp might offer tax advantages.",
            },
            Turn {
                role: "user",
                content: "James suggested converting to an S-Corp next year for self-employment tax savings.",
            },
            Turn {
                role: "assistant",
                content: "At 30k revenue, S-Corp savings can be meaningful.",
            },
            Turn {
                role: "user",
                content: "I track all consulting expenses with a separate credit card.",
            },
            Turn {
                role: "assistant",
                content: "Clean separation simplifies tax time.",
            },
            Turn {
                role: "user",
                content: "My home office deduction covers 150 square feet of my 2400 sq ft house.",
            },
            Turn {
                role: "assistant",
                content: "About 6 percent, which is a reasonable deduction.",
            },
            Turn {
                role: "user",
                content: "I deduct my internet, phone, and software subscriptions proportionally.",
            },
            Turn {
                role: "assistant",
                content: "All legitimate business deductions.",
            },
            Turn {
                role: "user",
                content: "The MacBook Pro was fully deducted as business equipment.",
            },
            Turn {
                role: "assistant",
                content: "Section 179 makes equipment deductions simple.",
            },
        ],
    },
    Session {
        id: "noise_34",
        turns: &[
            Turn {
                role: "user",
                content: "We had a neighborhood block party last Saturday.",
            },
            Turn {
                role: "assistant",
                content: "Community events are wonderful.",
            },
            Turn {
                role: "user",
                content: "About 30 families attended from our street.",
            },
            Turn {
                role: "assistant",
                content: "Strong neighborhood turnout.",
            },
            Turn {
                role: "user",
                content: "I grilled burgers and veggie patties for the whole block.",
            },
            Turn {
                role: "assistant",
                content: "The community grill master.",
            },
            Turn {
                role: "user",
                content: "Emma made her famous mango habanero salsa.",
            },
            Turn {
                role: "assistant",
                content: "That sounds delicious.",
            },
            Turn {
                role: "user",
                content: "Sofia played with the neighborhood kids until 9pm which is late for her.",
            },
            Turn {
                role: "assistant",
                content: "Special occasions deserve extended bedtimes.",
            },
            Turn {
                role: "user",
                content: "Our neighbors the Garcias brought tamales — incredible.",
            },
            Turn {
                role: "assistant",
                content: "Homemade tamales are special.",
            },
            Turn {
                role: "user",
                content: "We organized a neighborhood signal chat for coordinating events.",
            },
            Turn {
                role: "assistant",
                content: "Signal for neighborhood comms is privacy-conscious.",
            },
            Turn {
                role: "user",
                content: "There's interest in doing a quarterly block cleanup too.",
            },
            Turn {
                role: "assistant",
                content: "Community maintenance builds neighborhood pride.",
            },
            Turn {
                role: "user",
                content: "I volunteered to organize the January cleanup.",
            },
            Turn {
                role: "assistant",
                content: "Good citizenship.",
            },
            Turn {
                role: "user",
                content: "Biscuit was the star of the party, as usual.",
            },
            Turn {
                role: "assistant",
                content: "Golden retrievers always steal the show.",
            },
        ],
    },
    Session {
        id: "noise_35",
        turns: &[
            Turn {
                role: "user",
                content: "I'm considering getting LASIK eye surgery.",
            },
            Turn {
                role: "assistant",
                content: "LASIK has a high success rate for most prescriptions.",
            },
            Turn {
                role: "user",
                content: "I've worn glasses since I was 12, currently -3.5 in both eyes.",
            },
            Turn {
                role: "assistant",
                content: "A moderate prescription that typically responds well to LASIK.",
            },
            Turn {
                role: "user",
                content: "I had a consultation at Providence Eye Center.",
            },
            Turn {
                role: "assistant",
                content: "A reputable eye care provider.",
            },
            Turn {
                role: "user",
                content: "The ophthalmologist said I'm a good candidate.",
            },
            Turn {
                role: "assistant",
                content: "That's encouraging.",
            },
            Turn {
                role: "user",
                content: "The cost is about 4500 dollars for both eyes.",
            },
            Turn {
                role: "assistant",
                content: "A worthwhile investment given years of glasses costs.",
            },
            Turn {
                role: "user",
                content: "Emma had LASIK five years ago and loves the results.",
            },
            Turn {
                role: "assistant",
                content: "A trusted recommendation from someone you know.",
            },
            Turn {
                role: "user",
                content: "I'm thinking of scheduling it for after the Portland Marathon.",
            },
            Turn {
                role: "assistant",
                content: "Smart to avoid the recovery period during heavy training.",
            },
            Turn {
                role: "user",
                content: "The recovery involves avoiding swimming for 2 weeks.",
            },
            Turn {
                role: "assistant",
                content: "That confirms timing it after the marathon makes sense.",
            },
            Turn {
                role: "user",
                content: "I would love to see clearly without glasses for the first time in 22 years.",
            },
            Turn {
                role: "assistant",
                content: "That clarity is life-changing according to most patients.",
            },
            Turn {
                role: "user",
                content: "I'll decide after the marathon and back pain fully resolves.",
            },
            Turn {
                role: "assistant",
                content: "A prudent timeline.",
            },
        ],
    },
    Session {
        id: "noise_36",
        turns: &[
            Turn {
                role: "user",
                content: "I attended a conference on distributed systems in Seattle last week.",
            },
            Turn {
                role: "assistant",
                content: "A valuable conference for your work on Prometheus.",
            },
            Turn {
                role: "user",
                content: "The keynote was by Leslie Lamport on formal verification.",
            },
            Turn {
                role: "assistant",
                content: "Lamport is legendary in distributed systems.",
            },
            Turn {
                role: "user",
                content: "I sat in on a talk about Raft consensus that was directly relevant.",
            },
            Turn {
                role: "assistant",
                content: "Good timing given Kenji's work on the consensus layer.",
            },
            Turn {
                role: "user",
                content: "I took 15 pages of notes in my Obsidian vault.",
            },
            Turn {
                role: "assistant",
                content: "Good capture for future reference.",
            },
            Turn {
                role: "user",
                content: "I had dinner with David Park who drove down from Neurovex's Seattle office.",
            },
            Turn {
                role: "assistant",
                content: "Your former classmate.",
            },
            Turn {
                role: "user",
                content: "He confirmed the VP offer is still on the table for 6 months.",
            },
            Turn {
                role: "assistant",
                content: "A standing offer gives you flexibility.",
            },
            Turn {
                role: "user",
                content: "I used the trip to visit my mother Margaret in Seattle.",
            },
            Turn {
                role: "assistant",
                content: "Combining professional and family visits.",
            },
            Turn {
                role: "user",
                content: "She made her famous pot roast — I always request it when I visit.",
            },
            Turn {
                role: "assistant",
                content: "A classic comfort food from home.",
            },
            Turn {
                role: "user",
                content: "I stayed overnight at her place in Bellevue.",
            },
            Turn {
                role: "assistant",
                content: "Quality time with family.",
            },
            Turn {
                role: "user",
                content: "The train back from Seattle to Portland takes about 4 hours.",
            },
            Turn {
                role: "assistant",
                content: "The Amtrak Cascades is scenic.",
            },
        ],
    },
    Session {
        id: "noise_37",
        turns: &[
            Turn {
                role: "user",
                content: "I'm planning to apply for a US passport renewal — it expired.",
            },
            Turn {
                role: "assistant",
                content: "Allow 6-8 weeks for standard renewal.",
            },
            Turn {
                role: "user",
                content: "I want it renewed before the Costa Rica trip in December.",
            },
            Turn {
                role: "assistant",
                content: "You have enough time with expedited processing.",
            },
            Turn {
                role: "user",
                content: "Sofia needs her first passport too for the international travel.",
            },
            Turn {
                role: "assistant",
                content: "Child passports require both parents present for the application.",
            },
            Turn {
                role: "user",
                content: "We have an appointment at the Portland passport office next Tuesday.",
            },
            Turn {
                role: "assistant",
                content: "Good to have it scheduled.",
            },
            Turn {
                role: "user",
                content: "Emma's passport is still valid until 2027.",
            },
            Turn {
                role: "assistant",
                content: "She's covered.",
            },
            Turn {
                role: "user",
                content: "I gather the required photos and documents ahead of time.",
            },
            Turn {
                role: "assistant",
                content: "Preparation makes the appointment smooth.",
            },
            Turn {
                role: "user",
                content: "Sofia is excited about getting her first passport photo.",
            },
            Turn {
                role: "assistant",
                content: "A milestone for a 5-year-old.",
            },
            Turn {
                role: "user",
                content: "We need to leave at least 6 weeks before the December 20th departure.",
            },
            Turn {
                role: "assistant",
                content: "Apply by November 8th at the latest for standard processing.",
            },
            Turn {
                role: "user",
                content: "I'll choose expedited just to be safe — costs 60 dollars more.",
            },
            Turn {
                role: "assistant",
                content: "Worth it for peace of mind.",
            },
            Turn {
                role: "user",
                content: "I'll track the application status online.",
            },
            Turn {
                role: "assistant",
                content: "The State Department tracking is reliable.",
            },
        ],
    },
    Session {
        id: "noise_38",
        turns: &[
            Turn {
                role: "user",
                content: "I started a meditation practice — 10 minutes every morning.",
            },
            Turn {
                role: "assistant",
                content: "Consistent short sessions are more effective than occasional long ones.",
            },
            Turn {
                role: "user",
                content: "I use the Waking Up app by Sam Harris.",
            },
            Turn {
                role: "assistant",
                content: "Waking Up has excellent guided content.",
            },
            Turn {
                role: "user",
                content: "I meditate right after piano practice and before my run.",
            },
            Turn {
                role: "assistant",
                content: "A solid morning ritual stack.",
            },
            Turn {
                role: "user",
                content: "I've been doing it for 45 consecutive days.",
            },
            Turn {
                role: "assistant",
                content: "45 days is enough for it to feel like a habit.",
            },
            Turn {
                role: "user",
                content: "The effect on my focus and stress has been noticeable.",
            },
            Turn {
                role: "assistant",
                content: "Reduced reactivity is often the first noticeable benefit.",
            },
            Turn {
                role: "user",
                content: "Emma joined me for morning meditation last week.",
            },
            Turn {
                role: "assistant",
                content: "Shared practices can deepen relationships.",
            },
            Turn {
                role: "user",
                content: "I'm interested in eventually doing a silent meditation retreat.",
            },
            Turn {
                role: "assistant",
                content: "Ten days of silence is a profound experience.",
            },
            Turn {
                role: "user",
                content: "Dr. Wu my physiotherapist says mindfulness helps with chronic pain.",
            },
            Turn {
                role: "assistant",
                content: "The mind-body connection in pain management is well-documented.",
            },
            Turn {
                role: "user",
                content: "I've been sharing meditation tips on my blog.",
            },
            Turn {
                role: "assistant",
                content: "A natural extension of your knowledge-sharing.",
            },
            Turn {
                role: "user",
                content: "The meditation post got more traffic than my usual technical writing.",
            },
            Turn {
                role: "assistant",
                content: "Wellness content has broad appeal.",
            },
        ],
    },
    Session {
        id: "noise_39",
        turns: &[
            Turn {
                role: "user",
                content: "My team passed the code review milestone for Prometheus this week.",
            },
            Turn {
                role: "assistant",
                content: "A significant checkpoint.",
            },
            Turn {
                role: "user",
                content: "We achieved the sub-millisecond P99 latency target in testing.",
            },
            Turn {
                role: "assistant",
                content: "Excellent result.",
            },
            Turn {
                role: "user",
                content: "The Kafka integration is working end-to-end in the staging environment.",
            },
            Turn {
                role: "assistant",
                content: "A major integration milestone.",
            },
            Turn {
                role: "user",
                content: "Maria Santos delivered exceptional work on the Kafka component.",
            },
            Turn {
                role: "assistant",
                content: "Sounds like a key contributor to recognize.",
            },
            Turn {
                role: "user",
                content: "I gave her a spot bonus for the Kafka work.",
            },
            Turn {
                role: "assistant",
                content: "Timely recognition matters.",
            },
            Turn {
                role: "user",
                content: "We're now doing load testing at 500k events per second.",
            },
            Turn {
                role: "assistant",
                content: "An impressive throughput target.",
            },
            Turn {
                role: "user",
                content: "The Raft consensus layer added about 3 microseconds of overhead.",
            },
            Turn {
                role: "assistant",
                content: "Remarkably low overhead for consensus.",
            },
            Turn {
                role: "user",
                content: "We're on track for the November 1st deadline.",
            },
            Turn {
                role: "assistant",
                content: "Great progress given the earlier concerns.",
            },
            Turn {
                role: "user",
                content: "The load testing will continue through October 28th.",
            },
            Turn {
                role: "assistant",
                content: "Leaving three days for final fixes.",
            },
            Turn {
                role: "user",
                content: "I'm cautiously optimistic about a smooth launch.",
            },
            Turn {
                role: "assistant",
                content: "Cautious optimism is the right engineering mindset.",
            },
        ],
    },
    Session {
        id: "noise_40",
        turns: &[
            Turn {
                role: "user",
                content: "I'm planning a surprise birthday party for Emma next month.",
            },
            Turn {
                role: "assistant",
                content: "How exciting!",
            },
            Turn {
                role: "user",
                content: "Emma turns 33 on November 14th.",
            },
            Turn {
                role: "assistant",
                content: "November 14th is coming up.",
            },
            Turn {
                role: "user",
                content: "I'm booking a private room at her favorite restaurant Ava Gene's.",
            },
            Turn {
                role: "assistant",
                content: "Ava Gene's is excellent for a special occasion.",
            },
            Turn {
                role: "user",
                content: "I'll invite about 20 of her closest friends and family.",
            },
            Turn {
                role: "assistant",
                content: "A manageable size for a genuine surprise.",
            },
            Turn {
                role: "user",
                content: "Her best friend Zoe is helping me coordinate the guest list.",
            },
            Turn {
                role: "assistant",
                content: "A trusted accomplice for the surprise.",
            },
            Turn {
                role: "user",
                content: "I commissioned a custom portrait of our family from a local artist.",
            },
            Turn {
                role: "assistant",
                content: "A deeply personal gift.",
            },
            Turn {
                role: "user",
                content: "The artist is Maya Chen and her work is incredible.",
            },
            Turn {
                role: "assistant",
                content: "Having it commissioned well in advance was smart.",
            },
            Turn {
                role: "user",
                content: "I need to keep the party secret for 3 more weeks.",
            },
            Turn {
                role: "assistant",
                content: "The hardest part of a surprise party.",
            },
            Turn {
                role: "user",
                content: "Sofia knows but thinks it's a big adventure keeping the secret.",
            },
            Turn {
                role: "assistant",
                content: "Having a 5-year-old in on the secret is a wild card.",
            },
            Turn {
                role: "assistant",
                content: "Children have a special ability to blurt things out.",
            },
        ],
    },
    Session {
        id: "noise_41",
        turns: &[
            Turn {
                role: "user",
                content: "I just received my annual engineering compensation survey results.",
            },
            Turn {
                role: "assistant",
                content: "Always useful data.",
            },
            Turn {
                role: "user",
                content: "Senior architects in Portland make between 170 and 220k base.",
            },
            Turn {
                role: "assistant",
                content: "You're right at market with your 185k.",
            },
            Turn {
                role: "user",
                content: "The survey shows Rust skills command a 15 percent premium.",
            },
            Turn {
                role: "assistant",
                content: "Rust expertise is increasingly valued.",
            },
            Turn {
                role: "user",
                content: "I'll use this data in my November performance review with Dr. Huang.",
            },
            Turn {
                role: "assistant",
                content: "Market data is compelling negotiating context.",
            },
            Turn {
                role: "user",
                content: "I'm aiming for a 12 percent raise to 207k.",
            },
            Turn {
                role: "assistant",
                content: "Supported by market data and your Prometheus work.",
            },
            Turn {
                role: "user",
                content: "The Neurovex offer was at 240k base plus equity.",
            },
            Turn {
                role: "assistant",
                content: "Useful as reference data even though you declined.",
            },
            Turn {
                role: "user",
                content: "I want to stay at QuantumLogic but I need fair compensation.",
            },
            Turn {
                role: "assistant",
                content: "A reasonable position.",
            },
            Turn {
                role: "user",
                content: "Dr. Huang has always been fair about compensation.",
            },
            Turn {
                role: "assistant",
                content: "That's a good foundation for the conversation.",
            },
            Turn {
                role: "user",
                content: "I've documented all my contributions over the past year.",
            },
            Turn {
                role: "assistant",
                content: "A strong case built on evidence.",
            },
            Turn {
                role: "user",
                content: "Prometheus is my biggest leverage point in the review.",
            },
            Turn {
                role: "assistant",
                content: "A visible successful project is the best argument.",
            },
        ],
    },
    Session {
        id: "noise_42",
        turns: &[
            Turn {
                role: "user",
                content: "I got a speeding ticket on the way to the airport last Tuesday.",
            },
            Turn {
                role: "assistant",
                content: "That's frustrating.",
            },
            Turn {
                role: "user",
                content: "Doing 74 in a 65 on I-205.",
            },
            Turn {
                role: "assistant",
                content: "A moderate speed over the limit.",
            },
            Turn {
                role: "user",
                content: "The fine was 260 dollars.",
            },
            Turn {
                role: "assistant",
                content: "An expensive lesson in speed awareness.",
            },
            Turn {
                role: "user",
                content: "I'm going to fight it in traffic court.",
            },
            Turn {
                role: "assistant",
                content: "Worth attempting if you have time.",
            },
            Turn {
                role: "user",
                content: "Oregon allows a driving school option to waive the ticket.",
            },
            Turn {
                role: "assistant",
                content: "A good alternative to a fine and insurance impact.",
            },
            Turn {
                role: "user",
                content: "I'll do the online defensive driving course — 4 hours.",
            },
            Turn {
                role: "assistant",
                content: "4 hours online is manageable.",
            },
            Turn {
                role: "user",
                content: "The Rivian's adaptive cruise control is so good I forgot to check my speed.",
            },
            Turn {
                role: "assistant",
                content: "Technology dependency is a real driving hazard.",
            },
            Turn {
                role: "user",
                content: "I need to pay more attention to the speedometer on the highway.",
            },
            Turn {
                role: "assistant",
                content: "The Rivian's HUD shows speed prominently.",
            },
            Turn {
                role: "user",
                content: "Lesson learned — no more relying on cruise control near speed traps.",
            },
            Turn {
                role: "assistant",
                content: "An expensive but memorable lesson.",
            },
            Turn {
                role: "user",
                content: "Emma was not impressed when she found out.",
            },
            Turn {
                role: "assistant",
                content: "The domestic fallout from traffic violations is often the worst part.",
            },
        ],
    },
    Session {
        id: "noise_43",
        turns: &[
            Turn {
                role: "user",
                content: "I've been thinking about getting a second monitor for my home office.",
            },
            Turn {
                role: "assistant",
                content: "A second monitor significantly boosts productivity.",
            },
            Turn {
                role: "user",
                content: "I currently use a single LG 4K 27-inch with my MacBook.",
            },
            Turn {
                role: "assistant",
                content: "27-inch 4K is a quality setup already.",
            },
            Turn {
                role: "user",
                content: "I'm considering an ultra-wide 34-inch curved display.",
            },
            Turn {
                role: "assistant",
                content: "Ultra-wide is great for code and terminal side by side.",
            },
            Turn {
                role: "user",
                content: "The Dell U3423WE is highly reviewed for Mac compatibility.",
            },
            Turn {
                role: "assistant",
                content: "Dell U-series monitors are reliable.",
            },
            Turn {
                role: "user",
                content: "It has a built-in KVM switch which would help with my Mac Mini setup.",
            },
            Turn {
                role: "assistant",
                content: "KVM switches reduce cable clutter significantly.",
            },
            Turn {
                role: "user",
                content: "The cost is about 800 dollars — reasonable for the productivity gain.",
            },
            Turn {
                role: "assistant",
                content: "A monitor is one of the highest ROI workspace investments.",
            },
            Turn {
                role: "user",
                content: "My standing desk is the Uplift V2 with 72-inch top.",
            },
            Turn {
                role: "assistant",
                content: "Plenty of width for an ultra-wide setup.",
            },
            Turn {
                role: "user",
                content: "I ordered it on Amazon — arrives Thursday.",
            },
            Turn {
                role: "assistant",
                content: "Looking forward to the upgrade.",
            },
            Turn {
                role: "user",
                content: "I'll write a brief review on my blog once I've used it for a week.",
            },
            Turn {
                role: "assistant",
                content: "A helpful contribution for others considering it.",
            },
            Turn {
                role: "user",
                content: "Emma thinks I already have too many screens — she's not wrong.",
            },
            Turn {
                role: "assistant",
                content: "The classic 'more screens' marital dynamic.",
            },
        ],
    },
    Session {
        id: "noise_44",
        turns: &[
            Turn {
                role: "user",
                content: "I'm planning to do a digital detox weekend this month.",
            },
            Turn {
                role: "assistant",
                content: "A deliberate break from screens can be restorative.",
            },
            Turn {
                role: "user",
                content: "No phones or computers from Friday 6pm to Sunday 6pm.",
            },
            Turn {
                role: "assistant",
                content: "48 hours is a meaningful break.",
            },
            Turn {
                role: "user",
                content: "We'll go hiking at Silver Falls State Park.",
            },
            Turn {
                role: "assistant",
                content: "Silver Falls is stunning.",
            },
            Turn {
                role: "user",
                content: "The Trail of Ten Falls loop is one of the best hikes in Oregon.",
            },
            Turn {
                role: "assistant",
                content: "10 waterfalls in one hike is spectacular.",
            },
            Turn {
                role: "user",
                content: "Sofia is old enough now to do the 7-mile loop.",
            },
            Turn {
                role: "assistant",
                content: "At 5 she might need some carrying on the steep sections.",
            },
            Turn {
                role: "user",
                content: "I have an Osprey kid carrier backpack for those moments.",
            },
            Turn {
                role: "assistant",
                content: "Essential gear for hiking with young children.",
            },
            Turn {
                role: "user",
                content: "We'll stay at a rustic cabin in the park.",
            },
            Turn {
                role: "assistant",
                content: "On-site cabins make the experience complete.",
            },
            Turn {
                role: "user",
                content: "I'll bring books, a deck of cards, and board games.",
            },
            Turn {
                role: "assistant",
                content: "Analog entertainment for a screen-free weekend.",
            },
            Turn {
                role: "user",
                content: "The forecast is partly cloudy which is perfect hiking weather for Oregon.",
            },
            Turn {
                role: "assistant",
                content: "Better than rain or blazing sun.",
            },
            Turn {
                role: "user",
                content: "I'm genuinely excited to be unreachable for 48 hours.",
            },
            Turn {
                role: "assistant",
                content: "That excitement is a sign you need the break.",
            },
        ],
    },
    Session {
        id: "noise_45",
        turns: &[
            Turn {
                role: "user",
                content: "Kenji Tanaka submitted his resignation this morning.",
            },
            Turn {
                role: "assistant",
                content: "Unexpected news.",
            },
            Turn {
                role: "user",
                content: "He's been offered a remote position at a FAANG company.",
            },
            Turn {
                role: "assistant",
                content: "Hard to compete with FAANG compensation.",
            },
            Turn {
                role: "user",
                content: "His last day is November 15th — two weeks after the Prometheus launch.",
            },
            Turn {
                role: "assistant",
                content: "At least he'll see the project through.",
            },
            Turn {
                role: "user",
                content: "I'm sad to lose him — he's been on the team for four years.",
            },
            Turn {
                role: "assistant",
                content: "Four years is a long tenure in engineering.",
            },
            Turn {
                role: "user",
                content: "I need to accelerate knowledge transfer for the Raft layer.",
            },
            Turn {
                role: "assistant",
                content: "Documentation now is critical before he leaves.",
            },
            Turn {
                role: "user",
                content: "Kenji agreed to do two weeks of post-departure consulting at an hourly rate.",
            },
            Turn {
                role: "assistant",
                content: "Consulting bridge arrangements help with knowledge transitions.",
            },
            Turn {
                role: "user",
                content: "I'll open a senior engineer job requisition immediately.",
            },
            Turn {
                role: "assistant",
                content: "The sooner the process starts the better.",
            },
            Turn {
                role: "user",
                content: "The req will be Rust-required, distributed systems experience preferred.",
            },
            Turn {
                role: "assistant",
                content: "A specific and well-defined requirement.",
            },
            Turn {
                role: "user",
                content: "I posted to the tracewing Rust community — several interested candidates already.",
            },
            Turn {
                role: "assistant",
                content: "Your community presence pays off in recruiting.",
            },
            Turn {
                role: "user",
                content: "Losing both Liam and Kenji in the same period is challenging.",
            },
            Turn {
                role: "assistant",
                content: "A significant but manageable personnel challenge.",
            },
        ],
    },
    Session {
        id: "noise_46",
        turns: &[
            Turn {
                role: "user",
                content: "I need to renew my car registration for the Rivian by November 30th.",
            },
            Turn {
                role: "assistant",
                content: "Oregon vehicle registrations renew by the birthday month.",
            },
            Turn {
                role: "user",
                content: "I do everything online through the DMV website.",
            },
            Turn {
                role: "assistant",
                content: "The Oregon DMV online service is reliable.",
            },
            Turn {
                role: "user",
                content: "The Rivian has zero emissions so it qualifies for reduced registration fees.",
            },
            Turn {
                role: "assistant",
                content: "A small financial benefit of EV ownership.",
            },
            Turn {
                role: "user",
                content: "My driver's license also needs to be renewed — it expires in January.",
            },
            Turn {
                role: "assistant",
                content: "Oregon allows mail-in renewal for a limited number of cycles.",
            },
            Turn {
                role: "user",
                content: "I'll do both in one DMV visit to save time.",
            },
            Turn {
                role: "assistant",
                content: "Efficient.",
            },
            Turn {
                role: "user",
                content: "I have Real ID compliance on my current license already.",
            },
            Turn {
                role: "assistant",
                content: "Real ID is required for domestic flights starting May 2025.",
            },
            Turn {
                role: "user",
                content: "The Portland DMV at SE 122nd has shorter waits on Thursdays.",
            },
            Turn {
                role: "assistant",
                content: "Good tip.",
            },
            Turn {
                role: "user",
                content: "I'll go after Kenji's knowledge transfer session on Thursday.",
            },
            Turn {
                role: "assistant",
                content: "Efficient batching of errands.",
            },
            Turn {
                role: "user",
                content: "The whole thing should take under an hour.",
            },
            Turn {
                role: "assistant",
                content: "Oregon DMV has improved its wait times significantly.",
            },
            Turn {
                role: "user",
                content: "I'll bring work to review on my phone while waiting.",
            },
            Turn {
                role: "assistant",
                content: "Making the most of wait time.",
            },
        ],
    },
    Session {
        id: "noise_47",
        turns: &[
            Turn {
                role: "user",
                content: "I've been struggling with imposter syndrome on the Prometheus project.",
            },
            Turn {
                role: "assistant",
                content: "Imposter syndrome is extremely common among high achievers.",
            },
            Turn {
                role: "user",
                content: "Managing six engineers while still doing IC work is stretching me thin.",
            },
            Turn {
                role: "assistant",
                content: "The engineering manager-IC split is one of the hardest transitions.",
            },
            Turn {
                role: "user",
                content: "I talked to Dr. Huang about it and she was surprisingly supportive.",
            },
            Turn {
                role: "assistant",
                content: "Having a manager who understands is valuable.",
            },
            Turn {
                role: "user",
                content: "She suggested I reduce my IC contribution to focus more on leadership.",
            },
            Turn {
                role: "assistant",
                content: "Good advice for the Principal/VP level.",
            },
            Turn {
                role: "user",
                content: "I started seeing a therapist named Dr. Yolanda Torres on Thursdays.",
            },
            Turn {
                role: "assistant",
                content: "Therapy is a strength, not a weakness.",
            },
            Turn {
                role: "user",
                content: "Dr. Torres specializes in high-performing professionals.",
            },
            Turn {
                role: "assistant",
                content: "Specialized context makes therapy much more effective.",
            },
            Turn {
                role: "user",
                content: "The imposter syndrome has been with me since MIT graduate school.",
            },
            Turn {
                role: "assistant",
                content: "Long-standing patterns benefit from professional support.",
            },
            Turn {
                role: "user",
                content: "Meditation has helped me notice the self-critical thoughts more quickly.",
            },
            Turn {
                role: "assistant",
                content: "Mindfulness creates a gap between thought and reaction.",
            },
            Turn {
                role: "user",
                content: "I've made significant progress in just four sessions.",
            },
            Turn {
                role: "assistant",
                content: "Insight can come quickly with the right support.",
            },
            Turn {
                role: "user",
                content: "I want to be mentally strong for the senior principal promotion push.",
            },
            Turn {
                role: "assistant",
                content: "Mental fitness is as important as technical skill at that level.",
            },
        ],
    },
    Session {
        id: "noise_48",
        turns: &[
            Turn {
                role: "user",
                content: "I organized a team offsite for the Prometheus engineers.",
            },
            Turn {
                role: "assistant",
                content: "Offsites are great for team cohesion.",
            },
            Turn {
                role: "user",
                content: "We went to Timberline Lodge on Mount Hood for two days.",
            },
            Turn {
                role: "assistant",
                content: "Timberline is a classic Oregon venue.",
            },
            Turn {
                role: "user",
                content: "We did retrospectives in the morning and hiking in the afternoons.",
            },
            Turn {
                role: "assistant",
                content: "A good balance of work and team bonding.",
            },
            Turn {
                role: "user",
                content: "Maria Santos suggested a brilliant architecture simplification during the retro.",
            },
            Turn {
                role: "assistant",
                content: "The best ideas often come in relaxed settings.",
            },
            Turn {
                role: "user",
                content: "We cut three microservices that were adding unnecessary complexity.",
            },
            Turn {
                role: "assistant",
                content: "Simplification is often the hardest engineering decision.",
            },
            Turn {
                role: "user",
                content: "The team hiked the Mirror Lake trail on day two.",
            },
            Turn {
                role: "assistant",
                content: "Mirror Lake offers stunning views of Mount Hood.",
            },
            Turn {
                role: "user",
                content: "Even with Liam and Kenji leaving, morale is high.",
            },
            Turn {
                role: "assistant",
                content: "Visible leadership investment in the team pays off.",
            },
            Turn {
                role: "user",
                content: "QuantumLogic covered all expenses — about 4500 dollars total.",
            },
            Turn {
                role: "assistant",
                content: "Well-invested budget.",
            },
            Turn {
                role: "user",
                content: "Dr. Huang attended on the first day and had dinner with the team.",
            },
            Turn {
                role: "assistant",
                content: "VP presence at offsite signals organizational support.",
            },
            Turn {
                role: "user",
                content: "I'll continue doing quarterly offsites — the ROI is clear.",
            },
            Turn {
                role: "assistant",
                content: "Team cohesion compounds over time.",
            },
        ],
    },
    Session {
        id: "noise_49",
        turns: &[
            Turn {
                role: "user",
                content: "I'm getting ready for Sofia's fifth birthday party next weekend.",
            },
            Turn {
                role: "assistant",
                content: "An exciting milestone birthday!",
            },
            Turn {
                role: "user",
                content: "She requested a dinosaur theme party.",
            },
            Turn {
                role: "assistant",
                content: "Dinosaurs are eternally popular with kids.",
            },
            Turn {
                role: "user",
                content: "We're hosting 12 kids from her kindergarten class at our house.",
            },
            Turn {
                role: "assistant",
                content: "12 kindergartners — a brave choice.",
            },
            Turn {
                role: "user",
                content: "Emma ordered a custom T-rex cake from a bakery on Alberta Street.",
            },
            Turn {
                role: "assistant",
                content: "Custom cakes from local bakeries are special.",
            },
            Turn {
                role: "user",
                content: "I rented an inflatable bounce house for the backyard.",
            },
            Turn {
                role: "assistant",
                content: "A bounce house guarantees party success.",
            },
            Turn {
                role: "user",
                content: "Sofia's wish list includes a new bike with training wheels and a telescope.",
            },
            Turn {
                role: "assistant",
                content: "A telescope would complement her bird app interest.",
            },
            Turn {
                role: "user",
                content: "I got her a bright green bike with a matching helmet.",
            },
            Turn {
                role: "assistant",
                content: "A practical and exciting gift.",
            },
            Turn {
                role: "user",
                content: "Emma got her a beginners telescope — a Celestron StarSense Explorer.",
            },
            Turn {
                role: "assistant",
                content: "A perfect gift for a curious child.",
            },
            Turn {
                role: "user",
                content: "My mother Margaret is flying down from Seattle for the party.",
            },
            Turn {
                role: "assistant",
                content: "A whole family celebration.",
            },
            Turn {
                role: "user",
                content: "This is the first birthday party we've hosted since we moved to Portland.",
            },
            Turn {
                role: "assistant",
                content: "A milestone for the whole family.",
            },
        ],
    },
];

// ---------------------------------------------------------------------------
// Query definitions with ground-truth session IDs
// ---------------------------------------------------------------------------

const QUERIES: &[Query] = &[
    // Single-session queries
    Query {
        question: "What is the user's full name?",
        expected_sessions: &["personal_0"],
        query_type: "single-session",
    },
    Query {
        question: "What city does the user currently live in?",
        expected_sessions: &["personal_0"],
        query_type: "single-session",
    },
    Query {
        question: "What is the name of the user's wife?",
        expected_sessions: &["personal_0"],
        query_type: "single-session",
    },
    Query {
        question: "What is the user's daughter's name?",
        expected_sessions: &["personal_0"],
        query_type: "single-session",
    },
    Query {
        question: "What is the user's golden retriever's name?",
        expected_sessions: &["personal_0"],
        query_type: "single-session",
    },
    Query {
        question: "What company does the user work for?",
        expected_sessions: &["personal_1"],
        query_type: "single-session",
    },
    Query {
        question: "What is the user's job title at QuantumLogic?",
        expected_sessions: &["personal_1"],
        query_type: "single-session",
    },
    Query {
        question: "Who is the user's manager at QuantumLogic?",
        expected_sessions: &["personal_1"],
        query_type: "single-session",
    },
    Query {
        question: "What university did the user attend for their Master's degree?",
        expected_sessions: &["personal_1"],
        query_type: "single-session",
    },
    Query {
        question: "What is the user's absolute favorite cuisine?",
        expected_sessions: &["prefs_food_0"],
        query_type: "single-session",
    },
    Query {
        question: "What Thai dish does the user love most?",
        expected_sessions: &["prefs_food_0"],
        query_type: "single-session",
    },
    Query {
        question: "What dietary restriction does the user have regarding dairy?",
        expected_sessions: &["prefs_food_0"],
        query_type: "single-session",
    },
    Query {
        question: "What is the user's favorite Thai restaurant in Portland?",
        expected_sessions: &["prefs_food_0"],
        query_type: "single-session",
    },
    Query {
        question: "What music genre does the user prefer?",
        expected_sessions: &["prefs_music_0"],
        query_type: "single-session",
    },
    Query {
        question: "What is the user's all-time favorite jazz album?",
        expected_sessions: &["prefs_music_0"],
        query_type: "single-session",
    },
    Query {
        question: "What instrument does the user play?",
        expected_sessions: &["prefs_music_0"],
        query_type: "single-session",
    },
    Query {
        question: "What country did the user visit in March last year?",
        expected_sessions: &["travel_0"],
        query_type: "single-session",
    },
    Query {
        question: "Where did the user stay in Tokyo?",
        expected_sessions: &["travel_0"],
        query_type: "single-session",
    },
    Query {
        question: "What is the Prometheus project deadline at QuantumLogic?",
        expected_sessions: &["work_1"],
        query_type: "single-session",
    },
    Query {
        question: "What day is the user's one-on-one with Dr. Huang?",
        expected_sessions: &["work_0"],
        query_type: "single-session",
    },
    // Multi-session queries (cross-session reasoning)
    Query {
        question: "What did the user eat during their Tokyo trip given their food preferences?",
        expected_sessions: &["travel_0", "prefs_food_0"],
        query_type: "multi-session",
    },
    Query {
        question: "What piano piece might the user practice given their music preferences?",
        expected_sessions: &["prefs_music_0"],
        query_type: "multi-session",
    },
    Query {
        question: "How do the user's fitness habits relate to their marathon training?",
        expected_sessions: &["hobbies_0"],
        query_type: "multi-session",
    },
    Query {
        question: "How does the back pain affect the marathon training schedule?",
        expected_sessions: &["misc_13", "hobbies_0"],
        query_type: "multi-session",
    },
    Query {
        question: "What software tools does the user use for development and note-taking?",
        expected_sessions: &["misc_15", "misc_25"],
        query_type: "multi-session",
    },
    // Temporal ordering queries
    Query {
        question: "When was the Prometheus project deadline changed and what is the new date?",
        expected_sessions: &["work_1"],
        query_type: "temporal",
    },
    Query {
        question: "What happened with the Neurovex job offer and when was it resolved?",
        expected_sessions: &["misc_24"],
        query_type: "temporal",
    },
    Query {
        question: "When does the Oregon coast camping road trip happen versus the California trip?",
        expected_sessions: &["travel_1"],
        query_type: "temporal",
    },
    // Contradiction/update queries
    Query {
        question: "What is the current Prometheus deadline after the schedule change?",
        expected_sessions: &["work_1"],
        query_type: "knowledge-update",
    },
    Query {
        question: "Did the user accept or decline the VP Engineering offer at Neurovex?",
        expected_sessions: &["misc_24"],
        query_type: "knowledge-update",
    },
    // Personal context queries
    Query {
        question: "What side project is the user building in Rust?",
        expected_sessions: &["misc_19"],
        query_type: "single-session",
    },
    Query {
        question: "What is the user's marathon personal best time?",
        expected_sessions: &["hobbies_0"],
        query_type: "single-session",
    },
    Query {
        question: "What editor does the user use for Rust development?",
        expected_sessions: &["misc_15"],
        query_type: "single-session",
    },
    Query {
        question: "Where does the user's mother live?",
        expected_sessions: &["personal_0"],
        query_type: "single-session",
    },
    Query {
        question: "What is the user's brother's job?",
        expected_sessions: &["personal_0"],
        query_type: "single-session",
    },
];

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

struct QueryResult {
    hit1: bool,
    hit5: bool,
    hit10: bool,
    rr: Option<f64>,
    latency: Duration,
}

fn run_arena_bench() -> Vec<(String, Vec<QueryResult>)> {
    let dir = TempDir::new().expect("failed to create temp dir");
    let mut config = MemoryConfig::new(dir.path().join("arena.h5"), "arena-bench", EMBEDDING_DIM);
    config.wal_enabled = false;
    config.compact_threshold = 0.0;

    let mut memory = HDF5Memory::create(config).expect("failed to create HDF5Memory");

    // Ingest all sessions
    let mut entries: Vec<MemoryEntry> = Vec::new();
    let mut ts = 1_000_000.0f64;
    for session in SESSIONS {
        for turn in session.turns.iter() {
            entries.push(MemoryEntry {
                chunk: turn.content.to_string(),
                embedding: vec![0.0f32; EMBEDDING_DIM],
                source_channel: "arena".to_string(),
                timestamp: ts,
                session_id: session.id.to_string(),
                tags: String::new(),
            });
            ts += 1.0;
        }
    }
    memory.save_batch(entries).expect("failed to save entries");

    let zero_emb = vec![0.0f32; EMBEDDING_DIM];

    // Group results by query_type
    let mut by_type: std::collections::HashMap<String, Vec<QueryResult>> =
        std::collections::HashMap::new();

    for query in QUERIES {
        let expected: HashSet<&str> = query.expected_sessions.iter().copied().collect();

        let t0 = Instant::now();
        let results = memory.hybrid_search(&zero_emb, query.question, 0.0, 1.0, 10);
        let latency = t0.elapsed();

        let mut hit1 = false;
        let mut hit5 = false;
        let mut hit10 = false;
        let mut rr: Option<f64> = None;

        for (rank, result) in results.iter().enumerate() {
            let sess_id = memory.cache.session_ids[result.index].as_str();
            if expected.contains(sess_id) {
                hit10 = true;
                if rank < 5 {
                    hit5 = true;
                }
                if rank == 0 {
                    hit1 = true;
                }
                if rr.is_none() {
                    rr = Some(1.0 / (rank + 1) as f64);
                }
                break;
            }
        }

        let entry = by_type.entry(query.query_type.to_string()).or_default();
        entry.push(QueryResult {
            hit1,
            hit5,
            hit10,
            rr,
            latency,
        });
    }

    by_type.into_iter().collect()
}

fn print_results(results: &[(String, Vec<QueryResult>)]) {
    let total_turns: usize = SESSIONS
        .iter()
        .map(|s| s.turns.len())
        .collect::<Vec<_>>()
        .iter()
        .sum();
    println!("================================================================");
    println!("  MemoryArena Multi-Session Benchmark");
    println!("================================================================");
    println!();
    println!("Configuration:");
    println!(
        "  Sessions:      {} (deterministic seed data)",
        SESSIONS.len()
    );
    println!("  Total turns:   {}", total_turns);
    println!("  Queries:       {}", QUERIES.len());
    println!("  Embedding dim: {EMBEDDING_DIM} (zero vectors, BM25-only)");
    println!("  Topics: personal info, food preferences, music, travel, work/schedule, hobbies");
    println!();

    let mut all_hit1 = 0u32;
    let mut all_hit5 = 0u32;
    let mut all_hit10 = 0u32;
    let mut all_rr = 0.0f64;
    let mut all_count = 0u32;
    let mut all_latency_ns: Vec<u64> = Vec::new();

    let mut sorted: Vec<(&String, &Vec<QueryResult>)> =
        results.iter().map(|item| (&item.0, &item.1)).collect();
    sorted.sort_by_key(|(t, _)| t.as_str());

    println!(
        "{:<20} {:>5}  {:>7}  {:>7}  {:>7}  {:>7}  {:>12}",
        "Query Type", "N", "Hit@1", "Hit@5", "Hit@10", "MRR", "Avg Lat"
    );
    println!("{}", "-".repeat(72));

    for (qtype, qresults) in &sorted {
        let n = qresults.len() as u32;
        let h1 = qresults.iter().filter(|r| r.hit1).count() as u32;
        let h5 = qresults.iter().filter(|r| r.hit5).count() as u32;
        let h10 = qresults.iter().filter(|r| r.hit10).count() as u32;
        let rr_sum: f64 = qresults.iter().filter_map(|r| r.rr).sum();
        let avg_lat_us = qresults
            .iter()
            .map(|r| r.latency.as_nanos() as f64 / 1000.0)
            .sum::<f64>()
            / n.max(1) as f64;

        println!(
            "{:<20} {:>5}  {:>6.1}%  {:>6.1}%  {:>6.1}%  {:>7.4}  {:>9.1} µs",
            qtype,
            n,
            h1 as f64 / n.max(1) as f64 * 100.0,
            h5 as f64 / n.max(1) as f64 * 100.0,
            h10 as f64 / n.max(1) as f64 * 100.0,
            rr_sum / n.max(1) as f64,
            avg_lat_us
        );

        all_hit1 += h1;
        all_hit5 += h5;
        all_hit10 += h10;
        all_rr += rr_sum;
        all_count += n;
        for r in *qresults {
            all_latency_ns.push(r.latency.as_nanos() as u64);
        }
    }

    println!("{}", "-".repeat(72));
    all_latency_ns.sort_unstable();
    let p50_us = all_latency_ns
        .get(all_latency_ns.len() * 50 / 100)
        .copied()
        .unwrap_or(0) as f64
        / 1000.0;
    let p95_us = all_latency_ns
        .get(all_latency_ns.len() * 95 / 100)
        .copied()
        .unwrap_or(0) as f64
        / 1000.0;
    println!(
        "{:<20} {:>5}  {:>6.1}%  {:>6.1}%  {:>6.1}%  {:>7.4}",
        "OVERALL",
        all_count,
        all_hit1 as f64 / all_count.max(1) as f64 * 100.0,
        all_hit5 as f64 / all_count.max(1) as f64 * 100.0,
        all_hit10 as f64 / all_count.max(1) as f64 * 100.0,
        all_rr / all_count.max(1) as f64,
    );
    println!();
    println!(
        "Latency: p50={:.1} µs  p95={:.1} µs  ({} total turns in haystack)",
        p50_us, p95_us, total_turns
    );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("Running MemoryArena benchmark...");
    let results = run_arena_bench();
    println!();
    print_results(&results);
}
