import React from 'react';
import { createRoot } from 'react-dom/client';
import DemoGridFeatureCards from '@/components/demo-grid-feature-cards';
import DemoRuixenBentoCards from '@/components/demo-ruixen-bento-cards';
import DemoGridCard from '@/components/demo-grid-card';
import './index.css';

const container = document.getElementById('design-root');
if (!container) throw new Error('#design-root not found');

createRoot(container).render(
  <React.StrictMode>
    <div className="divide-y">
      <DemoGridFeatureCards />
      <DemoRuixenBentoCards />
      <DemoGridCard />
    </div>
  </React.StrictMode>,
);
