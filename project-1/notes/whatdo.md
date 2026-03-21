brief notes:

1. Znajdz dlugosc nukleotydow
2. Poszerz okno (6 - 14)
3. Znajdz 3 najwieksze pasujące
4. Clustering
5. znajdz wzorce conserwatywne (stumpy)
6. Wszystkie transkrypty zeby zidentyfikowac wszystkie miejsca wiazan

Step by step plan:

 - Phase 1: Preparation and Data Extraction
    1. Select your Protein: Choose either HNRNPC or HNRNPA2B1. This determines which dataset you will use.
    2. Define the Motif Length (w): * Open the expected_pattern file for your protein and count the number of nucleotides to find the base length (len).
         Set your search windows to w={len,len+1,len+2}. For example, if the expected motif is 6 nucleotides long, you will analyze windows of 6, 7, and 8.
    3. Extract Promising Motifs: * Go through the short fragments in the binding_sites_fshape.zip archive.
        Extract every continuous sequence of length w where at least one nucleotide has an fSHAPE reactivity value exceeding 1.0.

 - Phase 2: Pattern Discovery (Clustering)

    1. Run Cluster Analysis: * For each motif length (w) independently, perform clustering on the fSHAPE data profiles.
        Requirement: You must use at least two methods, such as K-Means++ and DBSCAN.
    2. Identify Consensus Motifs: * Look at the three most abundant clusters (each must have at least 3 members).
        Use a library like STUMPY to find the "consensus" fSHAPE profile (the representative pattern) for these clusters.

 - Phase 3: Transcript Searching and Validation
    1. Search the Transcripts: * Using the consensus profiles you found (and the original expected_pattern), search through the larger transcript files in the search_fshape.zip archive.
        Identify matches where the z-normalized Euclidean distance (znEd) is ≤ 2.5.
    2. Calculate Similarity Scores: * For every match found, calculate the Sequence Similarity Factor (ssf) based on the scoring system (2 pts for exact match, 1 pt for same type/purine-pyrimidine, 0 otherwise).
        Calculate the final General Similarity (aS) score: 
        aS=10⋅znEd−ssf.

 - Phase 4: Final Output
    1. Generate the Report Table: Create a table of your findings, ordered by the aS score (lowest/best first). Include:
        The motif sequence and its position (nucleotide range).
        The source filename.
        The calculated values for znEd, ssf, and aS.

