'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import toast, { Toaster } from 'react-hot-toast';
import { motion, AnimatePresence } from 'framer-motion';
import { SlidersHorizontal, ChevronDown } from 'lucide-react';
import { ProductCard, Product } from '@/components/ProductCard';
import { Header } from '@/components/Header';
import { SearchBar } from '@/components/SearchBar';
import { RecommendationSlider } from '@/components/RecommendationSlider';
import { ProductModal } from '@/components/ProductModal';

// Re-fetch recommendations only after this many buffered interactions.
// purchase = 2 pts, favorite = 1 pt, click = 0 pts (handled differently)
const REC_REFRESH_THRESHOLD = 3;

const SORT_OPTIONS = [
  { label: 'Featured', value: 'featured' },
  { label: 'Price: Low to High', value: 'price_asc' },
  { label: 'Price: High to Low', value: 'price_desc' },
  { label: 'AI Match', value: 'ai_score' },
];

function mockPrice(id: string) {
  return (parseInt(id.slice(-4)) % 150) + 49;
}

export default function Home() {
  const [userId, setUserId] = useState('user_1');

  // ── Main catalog / search ──────────────────────────────────────
  const [mainProducts, setMainProducts] = useState<Product[]>([]);
  const [mainLoading, setMainLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('featured');

  // ── Recommendations ────────────────────────────────────────────
  const [recommendations, setRecommendations] = useState<Product[]>([]);
  const [recLoading, setRecLoading] = useState(false);
  const [seedProduct, setSeedProduct] = useState<Product | null>(null);

  // ── Modal ──────────────────────────────────────────────────────
  const [modalProduct, setModalProduct] = useState<Product | null>(null);
  const [isSyncing, setIsSyncing] = useState(false);

  // ── Header counters ────────────────────────────────────────────
  const [cartCount, setCartCount] = useState(0);
  const [wishlistCount, setWishlistCount] = useState(0);

  // ── Smart recommendation refresh ───────────────────────────────
  // Accumulate weighted interactions; when threshold is reached we refresh recs.
  const interactionBuffer = useRef(0);
  const [pendingInteractions, setPendingInteractions] = useState(0);

  const isMounted = useRef(false);
  const recAbort = useRef<AbortController | null>(null);

  // ── Sort helper ────────────────────────────────────────────────
  const sortProducts = useCallback((items: Product[], by: string): Product[] => {
    if (by === 'ai_score') return [...items].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
    if (by === 'price_asc') return [...items].sort((a, b) => mockPrice(a.article_id) - mockPrice(b.article_id));
    if (by === 'price_desc') return [...items].sort((a, b) => mockPrice(b.article_id) - mockPrice(a.article_id));
    return items;
  }, []);

  // ── Fetch catalog / search results ────────────────────────────
  const fetchMain = useCallback(async (query: string) => {
    setMainLoading(true);
    try {
      let data: { results?: Product[] } = {};
      if (!query) {
        const res = await fetch('/api/catalog?limit=20');
        if (!res.ok) throw new Error();
        data = await res.json();
      } else {
        const res = await fetch('/api/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, mode: 'keyword', limit: 20 }),
        });
        if (!res.ok) throw new Error();
        data = await res.json();
      }
      setMainProducts(data.results ?? []);
    } catch {
      toast.error('Could not load products.', {
        style: { borderRadius: '100px', background: '#222', color: '#fff' },
      });
    } finally {
      setMainLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!isMounted.current) { fetchMain(''); isMounted.current = true; }
  }, [fetchMain]);

  // ── Fetch recommendations ──────────────────────────────────────
  const fetchRecommend = useCallback(async (articleId: string, seed: Product) => {
    if (recAbort.current) recAbort.current.abort();
    const ctrl = new AbortController();
    recAbort.current = ctrl;

    setSeedProduct(seed);
    setRecLoading(true);
    setPendingInteractions(0);
    interactionBuffer.current = 0;

    try {
      const res = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ article_id: articleId, user_id: userId, k: 10 }),
        signal: ctrl.signal,
      });
      if (!res.ok) throw new Error();
      const data = await res.json();
      setRecommendations(data.recommendations ?? []);
    } catch (e: unknown) {
      if (e instanceof Error && e.name !== 'AbortError') setRecommendations([]);
    } finally {
      if (!ctrl.signal.aborted) setRecLoading(false);
    }
  }, [userId]);

  // ── Core interaction handler ───────────────────────────────────
  /**
   * Logic mirrors real e-commerce platforms:
   *
   *  click       → Open product modal immediately + signal interact to backend.
   *                If this is a new seed (different product), fetch recommendations once.
   *
   *  favorite    → Optimistic wishlist counter + "Preference Updated!" toast.
   *                Buffer += 1. Re-fetch recs when buffer >= threshold.
   *
   *  purchase    → Optimistic cart counter + toast.
   *                Buffer += 2 (stronger signal). Re-fetch recs when buffer >= threshold.
   *
   * The buffer prevents a new recommendation call on every tap — recs only
   * refresh when the user has accumulated enough signal (like Zalando / ASOS).
   */
  const handleInteraction = useCallback(async (
    articleId: string,
    type: 'click' | 'favorite' | 'purchase',
  ) => {
    const allProducts = [...mainProducts, ...recommendations];
    const product = allProducts.find(p => p.article_id === articleId) ?? null;

    // ── Instant UI feedback ──────────────────────────────────────
    if (type === 'click') {
      setModalProduct(product);
      // Fetch recs only when the seed changes (user is exploring a new item)
      if (product && (!seedProduct || seedProduct.article_id !== articleId)) {
        fetchRecommend(articleId, product);
      }
      toast('Preference Updated!', {
        icon: '✨',
        style: { borderRadius: '100px', padding: '10px 20px', fontSize: '13px', fontWeight: 500 },
      });

    } else if (type === 'favorite') {
      setWishlistCount(c => c + 1);
      toast.success('Preference Updated!', {
        icon: '🤍',
        style: { borderRadius: '100px', padding: '10px 20px', fontSize: '13px', fontWeight: 500 },
      });
      interactionBuffer.current += 1;
      setPendingInteractions(Math.min(interactionBuffer.current, REC_REFRESH_THRESHOLD));

    } else if (type === 'purchase') {
      setCartCount(c => c + 1);
      toast.success('Preference Updated!', {
        icon: '🛍️',
        style: { borderRadius: '100px', padding: '10px 20px', fontSize: '13px', fontWeight: 500 },
      });
      interactionBuffer.current += 2;
      setPendingInteractions(Math.min(interactionBuffer.current, REC_REFRESH_THRESHOLD));
    }

    // ── Record signal with backend ───────────────────────────────
    setIsSyncing(true);
    fetch('/api/interact', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        article_id: articleId,
        interaction_type: type,
        shown_articles: allProducts.map(p => p.article_id),
      }),
    })
      .then(async res => {
        setIsSyncing(false);
        if (!res.ok) return;

        // After backend confirms: check if buffer has reached the threshold.
        // For favorite/purchase only — click already triggers recs above.
        if (type !== 'click' && interactionBuffer.current >= REC_REFRESH_THRESHOLD) {
          const currentSeed = seedProduct ?? product;
          if (currentSeed) {
            toast('Recommendations refreshed!', {
              icon: '🔄',
              style: { borderRadius: '100px', padding: '10px 20px', fontSize: '13px', fontWeight: 500 },
            });
            fetchRecommend(currentSeed.article_id, currentSeed);
          }
        }
      })
      .catch(() => setIsSyncing(false));
  }, [mainProducts, recommendations, seedProduct, userId, fetchRecommend]);

  // ── Search ─────────────────────────────────────────────────────
  const handleSearch = useCallback((query: string) => {
    setSearchQuery(query);
    fetchMain(query);
    if (!query) { setRecommendations([]); setSeedProduct(null); }
  }, [fetchMain]);

  // ── User switch ────────────────────────────────────────────────
  const handleUserChange = useCallback((id: string) => {
    setUserId(id);
    setRecommendations([]); setSeedProduct(null);
    setCartCount(0); setWishlistCount(0);
    interactionBuffer.current = 0; setPendingInteractions(0);
    fetchMain(searchQuery);
    toast(`Switched to ${id}`, {
      icon: '👤',
      style: { borderRadius: '100px', padding: '10px 20px', fontSize: '13px', fontWeight: 500 },
    });
  }, [fetchMain, searchQuery]);

  const sortedMain = sortProducts(mainProducts, sortBy);

  return (
    <div className="min-h-screen bg-white text-black font-sans selection:bg-black selection:text-white">
      <Toaster position="bottom-center" toastOptions={{ duration: 2500 }} />

      {/* ── Product Detail Modal ─────────────────────────────── */}
      <ProductModal
        product={modalProduct}
        onClose={() => setModalProduct(null)}
        onFavorite={id => handleInteraction(id, 'favorite')}
        onCart={id => handleInteraction(id, 'purchase')}
        isSyncing={isSyncing}
      />

      <Header
        userId={userId}
        setUserId={handleUserChange}
        cartCount={cartCount}
        wishlistCount={wishlistCount}
        isPersonalized={recommendations.length > 0}
      />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-10 pb-32">

        {/* ── Search ───────────────────────────────────────── */}
        <div className="mb-10">
          <SearchBar onSearch={handleSearch} loading={mainLoading} />
        </div>

        {/* ── Section title + Sort ─────────────────────────── */}
        <div className="flex items-end justify-between mb-7">
          <div>
            <AnimatePresence mode="wait">
              <motion.h2
                key={searchQuery || 'default'}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.22 }}
                className="text-2xl sm:text-3xl font-light tracking-tight text-gray-900"
              >
                {searchQuery ? `"${searchQuery}"` : 'The Edit'}
              </motion.h2>
            </AnimatePresence>
            <p className="text-xs text-gray-400 mt-1.5 tracking-wide">
              {mainLoading ? 'Loading…' : `${mainProducts.length} items`}
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button className="hidden sm:flex items-center gap-1.5 text-xs font-medium text-gray-600 hover:text-black px-3 py-2 border border-gray-200 rounded-full transition-colors">
              <SlidersHorizontal size={13} strokeWidth={1.5} />
              Filter
            </button>
            <div className="relative">
              <select
                className="appearance-none text-xs font-medium text-gray-600 bg-white pl-3 pr-7 py-2 border border-gray-200 rounded-full outline-none cursor-pointer hover:border-gray-400 transition-colors"
                value={sortBy}
                onChange={e => setSortBy(e.target.value)}
              >
                {SORT_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
              </select>
              <ChevronDown size={11} strokeWidth={2} className="absolute right-2.5 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none" />
            </div>
          </div>
        </div>

        {/* ── Main Product Grid ─────────────────────────────── */}
        {mainLoading && mainProducts.length === 0 ? (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-x-4 gap-y-12">
            {[...Array(8)].map((_, i) => (
              <div key={i} className="animate-pulse flex flex-col gap-4">
                <div className="bg-gray-100 aspect-[3/4] rounded-md" />
                <div className="space-y-2.5">
                  <div className="h-2.5 bg-gray-100 rounded w-3/4" />
                  <div className="h-2.5 bg-gray-100 rounded w-1/4" />
                </div>
              </div>
            ))}
          </div>
        ) : mainProducts.length === 0 ? (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="text-center py-28 flex flex-col items-center gap-4"
          >
            <p className="text-lg font-light text-gray-700">No items found for &ldquo;{searchQuery}&rdquo;</p>
            <button onClick={() => handleSearch('')}
              className="text-sm font-medium text-gray-500 hover:text-black underline underline-offset-4 transition-colors"
            >
              Clear search
            </button>
          </motion.div>
        ) : (
          <motion.div layout className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-x-4 gap-y-12">
            <AnimatePresence mode="popLayout">
              {sortedMain.map(p => (
                <ProductCard
                  key={p.article_id}
                  product={p}
                  onClick={id => handleInteraction(id, 'click')}
                  onFavorite={id => handleInteraction(id, 'favorite')}
                  onCart={id => handleInteraction(id, 'purchase')}
                />
              ))}
            </AnimatePresence>
          </motion.div>
        )}

        {/* ── Recommendation Slider ─────────────────────────── */}
        <AnimatePresence>
          {(recLoading || recommendations.length > 0) && (
            <RecommendationSlider
              products={recommendations}
              loading={recLoading}
              seedProduct={seedProduct}
              pendingInteractions={pendingInteractions}
              onInteract={handleInteraction}
            />
          )}
        </AnimatePresence>

      </main>
    </div>
  );
}
