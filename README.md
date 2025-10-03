# Smart_tickets_clustering
Smart_tickets_clustering is a text-based analysis tool designed to identify and group similar tickets, enabling automation opportunities.


Contexte : 
À mon arrivée au conseil départemental, j’ai observé une certaine récurrence dans les actions réalisées par les intégrateurs applicatifs. Cependant, il est difficile d’identifier précisément quelles actions sont suffisamment répétitives pour justifier une analyse approfondie (en vue d’une éventuelle automatisation). Cette difficulté s’explique par le volume important des demandes, la complexité des besoins métiers — souvent formulés différemment selon les interlocuteurs et renseignés de manière hétérogène dans notre outil de ticketing — ainsi que par la diversité de notre parc applicatif, qui compte plus de 150 applications. Ces éléments rendent tout regroupement difficile à obtenir directement via notre outil.
J’ai donc procédé à une extraction Excel des demandes métiers partageant les mêmes champs, afin de lancer une analyse textuelle ciblée sur les colonnes "Titre" et "Description", jugées les plus significatives à ce niveau.
Résultats obtenus :
- Les demandes ont été triées par clusters en fonction de la similarité de leur contenu.
- L’analyse a permis d’identifier 26 regroupements cohérents.
- Parmi ces 26 clusters, 8 tâches ont pu être automatisées.
