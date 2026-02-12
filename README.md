# AI Fashion Search

Moteur de recherche visuel pour la mode, propulsé par [Fashion CLIP](https://huggingface.co/patrickjohncyh/fashion-clip) et [Weaviate](https://weaviate.io/).

Recherchez des vêtements par texte ou par image grâce à la recherche hybride (dense vectors + BM25).

## Fonctionnalités

- **Recherche par texte** — décrivez un vêtement en langage naturel ("black leather jacket", "robe d'été fleurie")
- **Recherche par image** — uploadez une photo pour trouver des articles similaires
- **Recherche intelligente** — parsing sémantique des requêtes avec filtres automatiques (couleur, catégorie, prix, genre, marque)
- **Chatbot guidé** — sélection du genre → contexte de style → recherche affinée
- **Panel admin** — gestion des exemples, FAQ, paramètres du site, logs avec géolocalisation
- **Fiche produit** — vue détaillée avec toutes les photos d'un produit

## Tech Stack

| Composant | Technologie |
|-----------|-------------|
| Embeddings | [Fashion CLIP](https://huggingface.co/patrickjohncyh/fashion-clip) (512d) |
| Vector DB | Weaviate 1.28 |
| Backend | FastAPI + Uvicorn |
| Dataset | [ASOS e-commerce](https://huggingface.co/datasets/UniqueData/asos-e-commerce-dataset) |
| Infra | GCP (VM CPU) + Google Colab (GPU) |
| Runtime | Python 3.11, Docker Compose |

## Architecture

```
┌─────────────────────┐         ┌──────────────────────┐
│   Google Colab (GPU) │         │      GCP VM (CPU)     │
│                     │         │                      │
│  Fashion CLIP       │ vectors │  Weaviate :8080      │
│  encode images  ────┼────────►│  (vector DB)         │
│  (batch GPU)        │         │                      │
│                     │         │  FastAPI app :8000    │
└─────────────────────┘         │  (search + inference) │
                                └──────────────────────┘
```

- **Colab** : indexation lourde (encodage de milliers d'images) avec GPU gratuit
- **GCP** : Weaviate + app FastAPI en CPU (l'inférence = 1 encoding par requête, rapide sur CPU)

## Démarrage rapide

### 1. GCP — Lancer Weaviate + app

```bash
cp .env.example .env
docker compose up -d
```

L'application est disponible sur `http://<IP_GCP>:8000`.

Ports à ouvrir dans le firewall GCP : **8000** (app), **8080** (Weaviate HTTP), **50051** (Weaviate gRPC).

### 2. Colab — Indexer le dataset avec GPU

Ouvrir [`notebooks/index_asos_colab.ipynb`](notebooks/index_asos_colab.ipynb) dans Google Colab :

1. Activer le runtime GPU (Runtime → Change runtime type → T4)
2. Renseigner `GCP_EXTERNAL_IP` = IP externe de ta VM
3. Exécuter toutes les cellules

Le notebook encode les images sur GPU Colab et pousse les vecteurs directement dans Weaviate sur GCP.

### 3. Alternative — Indexer en CPU sur la VM

Pour un petit volume sans Colab :

```bash
docker compose run --rm indexer
```

## Développement local (sans Docker)

```bash
# Installer les dépendances
pip install -e ".[dev]"

# Lancer Weaviate seul
docker compose up -d weaviate

# Indexer
python scripts/index_asos.py --max-items 100

# Lancer le serveur
uvicorn src.web.app:app --reload --port 8000
```

## API

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/` | Interface web |
| `POST` | `/search` | Recherche texte (JSON: `query`, `gender`, `context`) |
| `POST` | `/search/image` | Recherche par image (multipart) |
| `GET` | `/featured` | Produits aléatoires pour la homepage |
| `GET` | `/product/{id}` | Détail d'un produit |
| `GET` | `/stats` | Statistiques de l'index |
| `GET` | `/health` | Health check |

## Structure du projet

```
├── src/
│   ├── config.py              # Configuration (env vars)
│   ├── models/
│   │   ├── base.py            # Interface modèle
│   │   └── clip_model.py      # Fashion CLIP wrapper
│   ├── database/
│   │   └── weaviate_client.py # Client Weaviate
│   ├── search/
│   │   ├── engine.py          # Moteur de recherche hybride
│   │   └── query_parser.py    # Parsing sémantique des requêtes
│   └── web/
│       ├── app.py             # Application FastAPI
│       ├── routes.py          # Routes API + admin
│       └── templates/
│           └── index.html     # Interface frontend
├── scripts/
│   └── index_asos.py          # Pipeline d'ingestion ASOS
├── notebooks/
│   └── index_asos_colab.ipynb # Indexation GPU via Google Colab
├── tests/
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

## Licence

Projet personnel de Julien G.
