MATCH p=(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'78621006'})
WHERE c1.FSN CONTAINS '(physical force)'
RETURN DISTINCT c1.sctid, c1.FSN;

