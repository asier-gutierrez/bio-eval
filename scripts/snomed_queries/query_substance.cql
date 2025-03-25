MATCH p=(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'105590001'})
WHERE c1.FSN CONTAINS '(substance)'
RETURN DISTINCT c1.sctid, c1.FSN;
