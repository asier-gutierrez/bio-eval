MATCH (d2: Description{active: '1', languageCode:'en'})<-[r3:HAS_DESCRIPTION]-(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'123037004'})
WHERE c1.FSN CONTAINS '(body structure)'
RETURN '(body structure)', COUNT(DISTINCT(d2.term));

MATCH (d2: Description{active: '1', languageCode:'en'})<-[r3:HAS_DESCRIPTION]-(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'373873005'})
WHERE c1.FSN CONTAINS '(clinical drug)'
RETURN '(clinical drug)', COUNT(DISTINCT(d2.term));

MATCH (d2: Description{active: '1', languageCode:'en'})<-[r3:HAS_DESCRIPTION]-(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'404684003'})
WHERE c1.FSN CONTAINS '(disorder)'
RETURN '(disorder)', COUNT(DISTINCT(d2.term));

MATCH (d2: Description{active: '1', languageCode:'en'})<-[r3:HAS_DESCRIPTION]-(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'308916002'})
WHERE c1.FSN CONTAINS '(environment)'
RETURN '(environment)', COUNT(DISTINCT(d2.term));

MATCH (d2: Description{active: '1', languageCode:'en'})<-[r3:HAS_DESCRIPTION]-(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'48176007'})
WHERE c1.FSN CONTAINS '(ethnic group)'
RETURN '(ethnic group)', COUNT(DISTINCT(d2.term));

MATCH (d2: Description{active: '1', languageCode:'en'})<-[r3:HAS_DESCRIPTION]-(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'272379006'})
WHERE c1.FSN CONTAINS '(event)'
RETURN '(event)', COUNT(DISTINCT(d2.term));

MATCH (d2: Description{active: '1', languageCode:'en'})<-[r3:HAS_DESCRIPTION]-(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'404684003'})
WHERE c1.FSN CONTAINS '(finding)'
RETURN '(finding)', COUNT(DISTINCT(d2.term));

MATCH (d2: Description{active: '1', languageCode:'en'})<-[r3:HAS_DESCRIPTION]-(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'308916002'})
WHERE c1.FSN CONTAINS '(geographic location)'
RETURN '(geographic location)', COUNT(DISTINCT(d2.term));

MATCH (d2: Description{active: '1', languageCode:'en'})<-[r3:HAS_DESCRIPTION]-(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'123037004'})
WHERE c1.FSN CONTAINS '(morphologic abnormality)'
RETURN '(morphologic abnormality)', COUNT(DISTINCT(d2.term));

MATCH (d2: Description{active: '1', languageCode:'en'})<-[r3:HAS_DESCRIPTION]-(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'363787002'})
WHERE c1.FSN CONTAINS '(observable entity)'
RETURN '(observable entity)', COUNT(DISTINCT(d2.term));

MATCH (d2: Description{active: '1', languageCode:'en'})<-[r3:HAS_DESCRIPTION]-(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'48176007'})
WHERE c1.FSN CONTAINS '(occupation)'
RETURN '(occupation)', COUNT(DISTINCT(d2.term));

MATCH (d2: Description{active: '1', languageCode:'en'})<-[r3:HAS_DESCRIPTION]-(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'410607006'})
WHERE c1.FSN CONTAINS '(organism)'
RETURN '(organism)', COUNT(DISTINCT(d2.term));

MATCH (d2: Description{active: '1', languageCode:'en'})<-[r3:HAS_DESCRIPTION]-(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'48176007'})
WHERE c1.FSN CONTAINS '(person)'
RETURN '(person)', COUNT(DISTINCT(d2.term));

MATCH (d2: Description{active: '1', languageCode:'en'})<-[r3:HAS_DESCRIPTION]-(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'373873005'})
WHERE c1.FSN CONTAINS '(product)'
RETURN '(product)', COUNT(DISTINCT(d2.term));

MATCH (d2: Description{active: '1', languageCode:'en'})<-[r3:HAS_DESCRIPTION]-(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'78621006'})
WHERE c1.FSN CONTAINS '(physical force)'
RETURN '(physical force)', COUNT(DISTINCT(d2.term));

MATCH (d2: Description{active: '1', languageCode:'en'})<-[r3:HAS_DESCRIPTION]-(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'260787004'})
WHERE c1.FSN CONTAINS '(physical object)'
RETURN '(physical object)', COUNT(DISTINCT(d2.term));

MATCH (d2: Description{active: '1', languageCode:'en'})<-[r3:HAS_DESCRIPTION]-(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'71388002'})
WHERE c1.FSN CONTAINS '(procedure)'
RETURN '(procedure)', COUNT(DISTINCT(d2.term));

MATCH (d2: Description{active: '1', languageCode:'en'})<-[r3:HAS_DESCRIPTION]-(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'362981000'})
WHERE c1.FSN CONTAINS '(qualifier value)'
RETURN '(qualifier value)', COUNT(DISTINCT(d2.term));

MATCH (d2: Description{active: '1', languageCode:'en'})<-[r3:HAS_DESCRIPTION]-(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'419891008'})
WHERE c1.FSN CONTAINS '(record artifact)'
RETURN '(record artifact)', COUNT(DISTINCT(d2.term));

MATCH (d2: Description{active: '1', languageCode:'en'})<-[r3:HAS_DESCRIPTION]-(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'71388002'})
WHERE c1.FSN CONTAINS '(regime/therapy)'
RETURN '(regime/therapy)', COUNT(DISTINCT(d2.term));

MATCH (d2: Description{active: '1', languageCode:'en'})<-[r3:HAS_DESCRIPTION]-(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'48176007'})
WHERE c1.FSN CONTAINS '(religion/philosophy)'
RETURN '(religion/philosophy)', COUNT(DISTINCT(d2.term));

MATCH (d2: Description{active: '1', languageCode:'en'})<-[r3:HAS_DESCRIPTION]-(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'243796009'})
WHERE c1.FSN CONTAINS '(situation)'
RETURN '(situation)', COUNT(DISTINCT(d2.term));

MATCH (d2: Description{active: '1', languageCode:'en'})<-[r3:HAS_DESCRIPTION]-(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'123038009'})
WHERE c1.FSN CONTAINS '(specimen)'
RETURN '(specimen)', COUNT(DISTINCT(d2.term));

MATCH (d2: Description{active: '1', languageCode:'en'})<-[r3:HAS_DESCRIPTION]-(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'105590001'})
WHERE c1.FSN CONTAINS '(substance)'
RETURN '(substance)', COUNT(DISTINCT(d2.term));








































