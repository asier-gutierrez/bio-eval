MATCH p=(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'272379006'})
WHERE c1.FSN CONTAINS '(event)'
RETURN DISTINCT c1.sctid, c1.FSN;

