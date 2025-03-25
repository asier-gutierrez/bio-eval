MATCH p=(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'308916002'})
WHERE c1.FSN CONTAINS '(environment)'
RETURN DISTINCT c1.sctid, c1.FSN;

