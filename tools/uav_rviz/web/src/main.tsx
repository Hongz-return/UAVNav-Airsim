import { createRoot } from 'react-dom/client';
import { App } from './App';
import { Connection } from './connection';
import { LiveStore } from './store';
import './styles.css';

const store = new LiveStore();
const connection = new Connection(store);
createRoot(document.getElementById('root')!).render(<App store={store} connection={connection} />);
