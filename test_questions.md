Here are some good test questions for your regulation RAG, especially if you stay with the **F1 Power Unit Financial Regulations** style corpus. I grouped them so you can test different abilities of the system.

## Basic retrieval questions

These check whether the system can find the right clause directly.

* What is the scope of the Power Unit Financial Regulations?
* What are the objectives of the Power Unit Financial Regulations?
* Who interprets and applies these regulations?
* When do these regulations come into force?
* What must a Power Unit Manufacturer do to demonstrate compliance with the Power Unit Cost Cap?
* What is the Power Unit Cost Cap for the N-3, N-2, and N-1 reporting periods?
* What is the Power Unit Cost Cap in a manufacturer’s inaugural season?
* What is a Reporting Group?
* What happens if a manufacturer has incurred less than 95% of Power Unit Activity costs itself?
* What costs are excluded under Article E3?

## More specific clause-level questions

These are better for testing whether the chunking is precise enough.

* Are marketing activities included in Relevant Costs?
* Are finance costs excluded from Relevant Costs?
* Are corporate income taxes excluded?
* Are employee bonus costs fully excluded, or is there a cap?
* Are health and safety costs excluded?
* Are transportation costs excluded?
* Are maternity and paternity leave costs excluded?
* Are hotel and flight costs for competitions excluded?
* Are heritage asset activities excluded from Relevant Costs?
* Are customer team power unit activities excluded?

## Questions requiring numerical details

These test whether the model can retrieve exact values rather than give a vague summary.

* What is the maximum amount of employee bonus costs that may be excluded?
* What is the cap on entertainment costs for employees?
* What amount is used for fuel purchased from a Fuel Supplier?
* What is the minimum value used for Single-Cylinder Dynamometer costs allocated to a Fuel Supplier?
* What downward adjustment applies to Eligible External Manufacturing Costs?
* What are the maximum unused cost cap adjustments for N-2 and N-1 reporting periods?

## Questions requiring interpretation across clauses

These are good for advanced RAG and multi-agent flow.

* If a cost relates partly to Marketing Activities and partly to Non-Power Unit Activities, how is it treated?
* How should costs be treated when one group entity recharges another for Power Unit Activities?
* How are inventories treated when calculating Relevant Costs?
* What happens if redundant inventories written off in a previous period are used later?
* How are foreign exchange transaction costs handled?
* How are related party transactions treated in Relevant Costs?
* What happens if research and development costs are deferred to a later reporting period?

## Cross-reference / multi-hop questions

These are especially useful if your advanced RAG includes a relevance judge or reference resolver.

* How does Article E4 adjust costs that were excluded under Article E3?
* What is the relationship between the Reporting Group rules in E2.4 and the Related Party Transaction adjustment in E4.1.1.a?
* If a Power Unit Manufacturer uses a presentation currency other than US Dollars, how is the Cost Cap determined and where is that addressed?
* Which exclusions in E3 interact with later clawback or adjustment provisions in E4?
* How do the regulations treat Non-Power Unit Activities across both exclusions and later adjustments?

## Good “hard negative” or ambiguity questions

These help test whether the system retrieves the right clause instead of something merely similar.

* Are legal activity costs excluded?
* Are property costs excluded?
* Are all employee-related costs excluded?
* Is all travel excluded?
* Are research and development costs excluded?
* Are fuel costs excluded?
* Are all taxes excluded?
* Is depreciation always excluded?

These are useful because the correct answer is often nuanced, not simply yes or no.

## End-to-end user-style questions

These feel more natural and are good for demoing the app.

* Can you summarize the main obligations of a Power Unit Manufacturer under these regulations?
* What are the main categories of excluded costs?
* Explain the Power Unit Cost Cap in simple terms.
* What kinds of adjustments can increase or decrease Relevant Costs?
* What should a new Power Unit Manufacturer know about these financial regulations?
* If I am unsure whether an entity belongs in the Reporting Group, what do the regulations say I should do?
* How do these regulations try to ensure fairness and financial sustainability?

## Very good evaluation set for your project

If you only want a compact test set of 10, I’d use these:

1. What are the objectives of the Power Unit Financial Regulations?
2. What is the Power Unit Cost Cap in a manufacturer’s inaugural season?
3. What is a Reporting Group?
4. Are employee bonus costs fully excluded, or is there a cap?
5. Are finance costs excluded from Relevant Costs?
6. How are Related Party Transactions treated in Relevant Costs?
7. How are inventories treated when calculating Relevant Costs?
8. What happens if research and development costs are deferred to a later reporting period?
9. Are hotel and flight costs for competitions excluded?
10. How do Article E3 exclusions interact with Article E4 adjustments?

If you want, I can turn these into a cleaner evaluation sheet with columns like question, expected article, expected clause, and difficulty.


## Messy / vague user-style questions (keywords not obvious)

* I'm trying to figure out whether I'm even in the right ruleset. Who is this framework meant for?
* When does this all become effective, like what "start date" should I use for planning?
* If someone challenges my interpretation, who is the authority that actually applies the rules?
* Can you explain the big picture goal of these rules without quoting the text?
* For a team in their first season, what kind of spending limit do they use, and how is it calculated?
* If my spending is split across different parts of the business, how do I decide what to ignore versus what to include?
* What are the typical categories of costs that usually get left out of the calculation?
* If an employee gets bonuses as part of a salary plan, is there a cap on what we can treat as excluded?
* Are staff travel and competition-related trips treated the same way as other business travel?
* What happens if a category of cost looks similar to something excluded - how strict is the treatment?

## Scenario-style questions (test retrieval with indirect wording)

* A constructor tells me they only kept a certain portion of their activity focused on the relevant program (less than "most of it"). What rule applies to them and what changes in their reporting?
* A team buys fuel from a third-party supplier but the price varies. How do the regulations handle the value used for that purchase in the cost calculation?
* An organization has a split operation where some work supports the racing program and some work supports everything else. What should they do when the expenses can't be neatly separated?
* A manufacturer uses an accounting approach where research & development is booked in one period but benefits another. How are those costs treated across reporting periods?
* During the season, a company charges "related parties" for certain services. How do the regulations treat those amounts when computing the figure that is subject to limits?
* A manufacturer excludes certain costs early on, but later there is a reconciliation mechanism that can adjust the excluded amounts. In practical terms, what kind of later adjustment can affect earlier exclusions?
* In a given reporting cycle, what should a team do if they're unsure which group(s) they belong to for the purposes of reporting?
* A team's budget includes entertainment and hospitality expenses for staff. Is every expense treated the same way, or is there a limit for what can be treated as excluded?
* A manufacturer has inventories on the books; later those inventories are written down. If they are reused later, what does the framework say to do?
* Someone asks whether legal, property, depreciation, and taxes all follow the same "left out" logic. What does the rule treatment look like at a category level?

