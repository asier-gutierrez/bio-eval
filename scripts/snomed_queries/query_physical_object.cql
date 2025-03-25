MATCH p=(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'260787004'})
WHERE c1.FSN CONTAINS '(physical object)'
RETURN DISTINCT c1.sctid, c1.FSN;

