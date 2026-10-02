# ImmoTunisia — Frontend

Interface React (Vite + TypeScript) combinant :
- **Mode Acheteur** : recommandation de propriétés via filtres et similarité cosinus
- **Mode Vendeur** : estimation de prix via modèle XGBoost

## Structure

```
src/
├── App.tsx                        # Routing principal + gestion du mode
├── constants.ts                   # Équipements, clés API
├── index.css                      # Variables CSS globales
├── main.tsx                       # Entry point React
├── components/
│   ├── Navbar.tsx / .module.css   # Barre de navigation Acheteur/Vendeur
│   ├── AmenityCheckboxes.tsx      # Grille d'équipements réutilisable
│   ├── FormField.tsx              # Label + input wrapper
│   └── shared.module.css          # Classes CSS partagées (inputs, grilles, boutons)
└── pages/
    ├── BuyerPage.tsx / .module.css   # Formulaire recommandation + résultats
    └── SellerPage.tsx / .module.css  # Formulaire prédiction + résultat prix
```

## Installation

```bash
npm install
npm run dev
```

L'app tourne sur http://localhost:5173

## Backend requis

Le frontend appelle deux endpoints :

| Endpoint | Méthode | Description |
|---|---|---|
| `http://127.0.0.1:8000/recommendation/recommend` | POST | Recommandations acheteur |
| `http://127.0.0.1:8000/predict` | POST | Estimation prix vendeur |

Assurez-vous que votre backend FastAPI tourne sur le port 8000.

## Changer l'URL du backend

Modifiez `src/constants.ts` :

```ts
export const API_BASE = 'http://127.0.0.1:8000'
```
