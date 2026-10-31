// Data from one recorded Forge run, read from the run's own files: events.jsonl,
// final-evaluation.json and the production scan. The page shows it in plain words.

/** exp-001 training loss, steps 1 to 96 (events.jsonl "metric"). The runtime was killed after step 76. */
export const LOSS = [
  0.4429, 0.6407, 0.8065, 0.6747, 0.5661, 0.7625, 0.3734, 0.6591, 0.5867,
  0.4861, 0.4565, 0.5998, 0.8453, 0.4141, 0.5478, 0.8212, 0.6998, 0.5881,
  0.3364, 0.7845, 0.7661, 0.5281, 0.4629, 0.6412, 0.5106, 1.411, 0.3024, 0.6292,
  0.3048, 0.1598, 0.8677, 0.2196, 0.4778, 0.344, 1.3852, 0.6713, 0.3262, 0.2411,
  0.1393, 0.3567, 0.444, 0.3431, 0.5201, 0.6445, 0.3371, 0.4128, 0.6704, 0.5012,
  0.6207, 0.4696, 0.8152, 0.1838, 0.3591, 0.2727, 0.4161, 0.461, 0.4003, 0.7729,
  0.6451, 0.5041, 1.1746, 0.4108, 0.3765, 0.6862, 0.4117, 0.4175, 0.2644,
  0.7988, 0.4118, 0.2108, 0.3199, 0.5888, 0.5869, 0.3717, 0.3079, 0.7323,
  0.4433, 0.8953, 0.548, 0.4031, 0.7762, 0.5723, 0.3453, 0.1987, 0.5476, 0.2813,
  0.2977, 0.8994, 0.5241, 0.9107, 0.5529, 0.4875, 0.4077, 0.379, 0.768, 0.3126,
];

/** Five of the answers Forge flagged in the agent's traffic (questions shortened). */
export const FLAGGED = [
  {
    question:
      "Willie came back from Europe with 70 euros. Normally 5 euros is worth 1\u2026",
    tags: ["Missing answer", "Cut off"],
  },
  {
    question:
      "The balance of Pam's bank account tripled during the year. At the end of the\u2026",
    tags: ["Math slip", "Missing answer", "Cut off"],
  },
  {
    question:
      "Fred was having a party and was responsible for buying canned soda.  He\u2026",
    tags: ["Missing answer", "Cut off"],
  },
  {
    question:
      "A math teacher had $100 to buy three different types of calculators. A basic\u2026",
    tags: ["Missing answer"],
  },
  {
    question:
      "Robbie weighs 100 pounds. Patty was 4.5 times as heavy as Robbie, but she has\u2026",
    tags: ["Missing answer", "Cut off"],
  },
];

export type Role = "data" | "director" | "trainer" | "evaluator" | "governance";
export const ROLES: Record<Role, { name: string; initials: string }> = {
  data: { name: "Data researcher", initials: "DR" },
  director: { name: "Research director", initials: "RD" },
  trainer: { name: "Training engineer", initials: "TE" },
  evaluator: { name: "Experiment evaluator", initials: "EV" },
  governance: { name: "Governance council", initials: "GC" },
};

export type Message = {
  from: Role;
  to: Role | "room";
  route: "Direct message" | "Research room" | "Reply in thread";
  text: string;
  receipt: string;
};

/** The run's Agent Relay messages, retold in plain words (same senders, routes and order). */
export const MESSAGES: Message[] = [
  {
    from: "data",
    to: "director",
    route: "Direct message",
    text: "I found the answers that need fixing. Half will teach the model, and half are saved to test it.",
    receipt: "Read by the director",
  },
  {
    from: "director",
    to: "room",
    route: "Research room",
    text: "Plan: teach the model to finish every answer with a clear final number.",
    receipt: "Read by the team",
  },
  {
    from: "trainer",
    to: "room",
    route: "Research room",
    text: "The new version is trained and saved for review.",
    receipt: "Read by the team",
  },
  {
    from: "evaluator",
    to: "director",
    route: "Direct message",
    text: "It now answers questions it used to miss, and keeps the ones it already knew.",
    receipt: "Read by the director",
  },
  {
    from: "governance",
    to: "room",
    route: "Reply in thread",
    text: "Approved. Every check passed.",
    receipt: "Read by the team",
  },
];

/** GSM8K test-885 from final-evaluation.json, verbatim. */
export const EXAMPLE = {
  id: "test-885",
  question:
    "The Ben and Aggie Burrito Shop makes 125 chimichangas on Tuesdays, 125 chimichangas on Wednesdays and twice as many on Friday. How many chimichangas do they make on those three days?",
  before:
    "To determine how many chimichangas the Ben and Aggie Burrito Shop makes on each of the three days (Tuesday, Wednesday, and Friday), we can follow these steps:\n\n1. Identify the number of chimichangas made on Tuesday.\n   - The shop makes 125 chimichangas on Tuesday.\n\n2. Identify the number of chimichangas made on Wednesday.\n   - The shop makes twice as many chimichangas on Wednesday as it does on Tuesday.\n   - Therefore, the number of chimichangas made on Wednesday is \\(2 \\times 125 = 250\\).\n\n3. Identify the number of chimichangas made on Friday.\n   - The shop makes 125 chimichangas on Friday.\n\n4. Sum up the total number of chimichangas made over the three days.\n   - Total chimichangas = Chimbachamas on Tuesday + Chimbachamas on Wednesday + Chimbachamas on Friday\n   - Total chimichangas = 125 + 250 + 125\n\nNow, let's perform the addition:\n\\[125 + 250 = 375\\]\n\\[",
  after:
    "On Tuesday, they make 125.\nOn Wednesday, they make 125.\nOn Friday, they make twice as many as on Tuesday, which is 2*125 = <<125*2=250>>250.\nIn total, they make 125+125+250 = <<125+125+250=500>>500 chimichangas.\n#### 500",
  /** Exact substrings to highlight, in order: where each answer goes wrong or right. */
  beforeMarks: [
    "The shop makes twice as many chimichangas on Wednesday as it does on Tuesday.",
    "\\[125 + 250 = 375\\]\n\\[",
  ],
  afterMarks: ["On Friday, they make twice as many as on Tuesday", "#### 500"],
};
