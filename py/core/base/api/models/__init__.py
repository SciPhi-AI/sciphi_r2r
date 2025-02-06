from shared.api.models.auth.responses import (
    TokenResponse,
    WrappedTokenResponse,
)
from shared.api.models.base import (
    GenericBooleanResponse,
    GenericMessageResponse,
    PaginatedR2RResult,
    R2RResults,
    WrappedBooleanResponse,
    WrappedGenericMessageResponse,
)
from shared.api.models.graph.responses import (  # TODO: Need to review anything above this
    Community,
    Entity,
    GraphResponse,
    Relationship,
    WrappedCommunitiesResponse,
    WrappedCommunityResponse,
    WrappedEntitiesResponse,
    WrappedEntityResponse,
    WrappedGraphResponse,
    WrappedGraphsResponse,
    WrappedRelationshipResponse,
    WrappedRelationshipsResponse,
)
from shared.api.models.ingestion.responses import (
    IngestionResponse,
    UpdateResponse,
    VectorIndexResponse,
    VectorIndicesResponse,
    WrappedIngestionResponse,
    WrappedMetadataUpdateResponse,
    WrappedUpdateResponse,
    WrappedVectorIndexResponse,
    WrappedVectorIndicesResponse,
)
from shared.api.models.management.responses import (  # Document Responses; Prompt Responses; Chunk Responses; Conversation Responses; User Responses; TODO: anything below this hasn't been reviewed
    AnalyticsResponse,
    ChunkResponse,
    CollectionResponse,
    ConversationResponse,
    LogResponse,
    PromptResponse,
    ServerStats,
    SettingsResponse,
    User,
    WrappedAnalyticsResponse,
    WrappedAPIKeyResponse,
    WrappedAPIKeysResponse,
    WrappedChunkResponse,
    WrappedChunksResponse,
    WrappedCollectionResponse,
    WrappedCollectionsResponse,
    WrappedConversationMessagesResponse,
    WrappedConversationResponse,
    WrappedConversationsResponse,
    WrappedDocumentResponse,
    WrappedDocumentsResponse,
    WrappedLogsResponse,
    WrappedMessageResponse,
    WrappedMessagesResponse,
    WrappedPromptResponse,
    WrappedPromptsResponse,
    WrappedResetDataResult,
    WrappedServerStatsResponse,
    WrappedSettingsResponse,
    WrappedUserResponse,
    WrappedUsersResponse,
    WrappedVerificationResult,
)
from shared.api.models.retrieval.responses import (
    AgentResponse,
    RAGResponse,
    WrappedAgentResponse,
    WrappedCompletionResponse,
    WrappedDocumentSearchResponse,
    WrappedRAGResponse,
    WrappedSearchResponse,
    WrappedVectorSearchResponse,
)

__all__ = [
    # Auth Responses
    "TokenResponse",
    "WrappedTokenResponse",
    "WrappedVerificationResult",
    "WrappedGenericMessageResponse",
    "WrappedResetDataResult",
    # Ingestion Responses
    "IngestionResponse",
    "WrappedIngestionResponse",
    "WrappedUpdateResponse",
    "WrappedMetadataUpdateResponse",
    "WrappedVectorIndexResponse",
    "WrappedVectorIndicesResponse",
    "UpdateResponse",
    "VectorIndexResponse",
    "VectorIndicesResponse",
    # Knowledge Graph Responses
    "Entity",
    "Relationship",
    "Community",
    "WrappedEntityResponse",
    "WrappedEntitiesResponse",
    "WrappedRelationshipResponse",
    "WrappedRelationshipsResponse",
    "WrappedCommunityResponse",
    "WrappedCommunitiesResponse",
    # TODO: Need to review anything above this
    "GraphResponse",
    "WrappedGraphResponse",
    "WrappedGraphsResponse",
    # Management Responses
    "PromptResponse",
    "ServerStats",
    "LogResponse",
    "AnalyticsResponse",
    "SettingsResponse",
    "ChunkResponse",
    "CollectionResponse",
    "WrappedServerStatsResponse",
    "WrappedLogsResponse",
    "WrappedAnalyticsResponse",
    "WrappedSettingsResponse",
    "WrappedDocumentResponse",
    "WrappedDocumentsResponse",
    "WrappedCollectionResponse",
    "WrappedCollectionsResponse",
    # Conversation Responses
    "ConversationResponse",
    "WrappedConversationMessagesResponse",
    "WrappedConversationResponse",
    "WrappedConversationsResponse",
    # Prompt Responses
    "WrappedPromptResponse",
    "WrappedPromptsResponse",
    # Conversation Responses
    "WrappedMessageResponse",
    "WrappedMessagesResponse",
    # Chunk Responses
    "WrappedChunkResponse",
    "WrappedChunksResponse",
    # User Responses
    "User",
    "WrappedUserResponse",
    "WrappedUsersResponse",
    "WrappedAPIKeyResponse",
    # Base Responses
    "PaginatedR2RResult",
    "R2RResults",
    "GenericBooleanResponse",
    "GenericMessageResponse",
    "WrappedBooleanResponse",
    "WrappedGenericMessageResponse",
    # TODO: This needs to be cleaned up
    "RAGResponse",
    "AgentResponse",
    "WrappedDocumentSearchResponse",
    "WrappedSearchResponse",
    "WrappedVectorSearchResponse",
    "WrappedCompletionResponse",
    "WrappedRAGResponse",
    "WrappedAgentResponse",
]
