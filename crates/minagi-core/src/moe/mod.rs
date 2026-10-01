//! The mixture-of-experts pool: bookkeeping ([`book`]), where experts live when not on the card ([`store`]), the card
//! itself ([`pool`]) and routing/dispatch ([`route`]).

pub mod book;
pub mod pool;
pub mod route;
pub mod store;
pub mod tiers_store;

pub use book::{PoolBook, SlotPlan};
pub use pool::{Lineage, PagedPool};
pub use route::{RouteCfg, RouteOut, RouteStats};
pub use store::{ExpertEntry, ExpertMoments, ExpertStore, MemStore, StoreReport};
pub use tiers_store::TiersStore;
