## NUS-WIDE status

NUS-WIDE cleaning is complete.

Completed parts:
1. canonical raw cleaning
2. ra-style top-10 category filtering based on canonical clean

Final counts:
- canonical clean rows: 269642
- ra-style filtered rows from canonical clean: 186571

Difference from ra reported 186577:
-6, caused by strict removal of 6 ambiguous duplicates.

Current rule:
- do not modify NUS-WIDE cleaning logic unless a clear bug is found

Current next step:
- plan common cleaning pipeline abstraction
- do not start training
- do not start feature extraction
- do not start MIRFlickr formal implementation yet