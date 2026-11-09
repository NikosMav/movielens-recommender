import {createRoot} from 'react-dom/client';
import {Theme} from '@astryxdesign/core/theme';
import {neutralTheme} from '@astryxdesign/theme-neutral/built';

import '@astryxdesign/core/reset.css';
import '@astryxdesign/core/astryx.css';
import '@astryxdesign/theme-neutral/theme.css';
import '@fontsource-variable/figtree';
import '@fontsource-variable/fraunces';
import './styles.css';
import {App} from './App';

createRoot(document.getElementById('root')!).render(
  <Theme theme={neutralTheme} mode="light">
    <App />
  </Theme>,
);
