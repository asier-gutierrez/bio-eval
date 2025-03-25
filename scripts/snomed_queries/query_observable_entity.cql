MATCH p=(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'363787002'})
WHERE c1.FSN CONTAINS '(observable entity)'
RETURN DISTINCT c1.sctid, c1.FSN;

