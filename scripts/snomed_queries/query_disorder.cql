MATCH p=(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'404684003'})
WHERE c1.FSN CONTAINS '(disorder)'
RETURN DISTINCT c1.sctid, c1.FSN;

