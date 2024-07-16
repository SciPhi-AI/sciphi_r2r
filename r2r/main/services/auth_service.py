from datetime import datetime, timedelta

from r2r.base import (
    KVLoggingSingleton,
    R2RException,
    RunManager,
    Token,
    User,
    UserCreate,
)
from r2r.telemetry.telemetry_decorator import telemetry_event

from ..abstractions import R2RPipelines, R2RProviders
from ..assembly.config import R2RConfig
from .base import Service


class AuthService(Service):
    def __init__(
        self,
        config: R2RConfig,
        providers: R2RProviders,
        pipelines: R2RPipelines,
        run_manager: RunManager,
        logging_connection: KVLoggingSingleton,
    ):
        super().__init__(
            config, providers, pipelines, run_manager, logging_connection
        )

    @telemetry_event("RegisterUser")
    async def register(self, user: UserCreate) -> User:
        return self.providers.auth.register(user)

    @telemetry_event("VerifyEmail")
    async def verify_email(self, verification_code: str) -> bool:

        if not self.config.auth.require_email_verification:
            raise R2RException(
                status_code=400, message="Email verification is not required"
            )

        user_id = self.providers.database.relational.get_user_id_by_verification_code(
            verification_code
        )
        if not user_id:
            raise R2RException(
                status_code=400, message="Invalid or expired verification code"
            )

        self.providers.database.relational.mark_user_as_verified(user_id)
        self.providers.database.relational.remove_verification_code(
            verification_code
        )
        return True

    @telemetry_event("Login")
    async def login(self, email: str, password: str) -> dict[str, Token]:
        return self.providers.auth.login(email, password)

    @telemetry_event("GetCurrentUser")
    async def user_info(self, token: str) -> User:
        token_data = self.providers.auth.decode_token(token)
        user = self.providers.database.relational.get_user_by_email(
            token_data.email
        )
        if user is None:
            raise R2RException(
                status_code=401, message="Invalid authentication credentials"
            )
        return user

    @telemetry_event("RefreshToken")
    async def refresh_access_token(
        self, user_email: str, refresh_token: str
    ) -> dict[str, Token]:
        return self.providers.auth.refresh_access_token(
            user_email, refresh_token
        )
