from typing import Any, Optional
import datetime
import decimal

from sqlalchemy import BigInteger, Date, Double, Enum, ForeignKeyConstraint, Index, Integer, String, Text
from sqlalchemy.dialects.mysql import BIT, DATETIME
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass


class Company(Base):
    __tablename__ = 'company'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    logo_path: Mapped[Optional[str]] = mapped_column(String(255))
    name: Mapped[Optional[str]] = mapped_column(String(255))

    movie_company: Mapped[list['MovieCompany']] = relationship('MovieCompany', back_populates='company')


class Genre(Base):
    __tablename__ = 'genre'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[Optional[str]] = mapped_column(String(255))

    movie_genre: Mapped[list['MovieGenre']] = relationship('MovieGenre', back_populates='genre')


class JwtRefreshEntity(Base):
    __tablename__ = 'jwt_refresh_entity'

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    refresh: Mapped[str] = mapped_column(String(512), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    create_at: Mapped[Optional[datetime.datetime]] = mapped_column(DATETIME(fsp=6))


class Movie(Base):
    __tablename__ = 'movie'
    __table_args__ = (
        Index('UKbgtoou3ammfcnlke2mv6uo522', 'tmdb_id', unique=True),
    )

    adult: Mapped[Any] = mapped_column(BIT(1), nullable=False)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    runtime: Mapped[int] = mapped_column(Integer, nullable=False)
    vote_average: Mapped[decimal.Decimal] = mapped_column(Double(asdecimal=True), nullable=False)
    tmdb_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    release_date: Mapped[Optional[datetime.date]] = mapped_column(Date)
    original_language: Mapped[Optional[str]] = mapped_column(String(255))
    overview: Mapped[Optional[str]] = mapped_column(Text)
    poster_path: Mapped[Optional[str]] = mapped_column(String(255))
    title: Mapped[Optional[str]] = mapped_column(String(255))

    movie_company: Mapped[list['MovieCompany']] = relationship('MovieCompany', back_populates='movie')
    movie_genre: Mapped[list['MovieGenre']] = relationship('MovieGenre', back_populates='movie')
    movie_people: Mapped[list['MoviePeople']] = relationship('MoviePeople', back_populates='movie')
    review: Mapped[list['Review']] = relationship('Review', back_populates='movie')
    user_movie: Mapped[list['UserMovie']] = relationship('UserMovie', back_populates='movie')


class People(Base):
    __tablename__ = 'people'
    __table_args__ = (
        Index('UKsi7j5i673c2k03dmusp8jub3n', 'tmdb_id', unique=True),
    )

    gender: Mapped[int] = mapped_column(Integer, nullable=False)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    birth_day: Mapped[Optional[datetime.date]] = mapped_column(Date)
    tmdb_id: Mapped[Optional[int]] = mapped_column(BigInteger)
    biography: Mapped[Optional[str]] = mapped_column(Text)
    name: Mapped[Optional[str]] = mapped_column(String(255))
    profile_image_path: Mapped[Optional[str]] = mapped_column(String(255))
    job: Mapped[Optional[str]] = mapped_column(Enum('ACTOR', 'PRODUCER'))

    movie_people: Mapped[list['MoviePeople']] = relationship('MoviePeople', back_populates='people')


class User(Base):
    __tablename__ = 'user'
    __table_args__ = (
        Index('UKgj2fy3dcix7ph7k8684gka40c', 'name', unique=True),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    is_lock: Mapped[Any] = mapped_column(BIT(1), nullable=False)
    is_social: Mapped[Any] = mapped_column(BIT(1), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    role_type: Mapped[str] = mapped_column(Enum('ADMIN', 'USER'), nullable=False)
    create_at: Mapped[Optional[datetime.date]] = mapped_column(Date)
    age_group: Mapped[Optional[str]] = mapped_column(String(255))
    location: Mapped[Optional[str]] = mapped_column(String(255))
    social_provider_type: Mapped[Optional[str]] = mapped_column(Enum('GOOGLE', 'NAVER'))
    gender: Mapped[Optional[str]] = mapped_column(String(255))
    provider_id: Mapped[Optional[str]] = mapped_column(String(255))

    user_movie: Mapped[list['UserMovie']] = relationship('UserMovie', back_populates='user')


class MovieCompany(Base):
    __tablename__ = 'movie_company'
    __table_args__ = (
        ForeignKeyConstraint(['company_id'], ['company.id'], name='FKpdv9s1gqh79ffs6ketmpjfqbo'),
        ForeignKeyConstraint(['movie_id'], ['movie.id'], name='FKel8kolqtcl37x5fxxvs4m15ev'),
        Index('FKel8kolqtcl37x5fxxvs4m15ev', 'movie_id'),
        Index('FKpdv9s1gqh79ffs6ketmpjfqbo', 'company_id')
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[Optional[int]] = mapped_column(Integer)
    movie_id: Mapped[Optional[int]] = mapped_column(Integer)

    company: Mapped[Optional['Company']] = relationship('Company', back_populates='movie_company')
    movie: Mapped[Optional['Movie']] = relationship('Movie', back_populates='movie_company')


class MovieGenre(Base):
    __tablename__ = 'movie_genre'
    __table_args__ = (
        ForeignKeyConstraint(['genre_id'], ['genre.id'], name='FK86p3roa187k12avqfl28klp1q'),
        ForeignKeyConstraint(['movie_id'], ['movie.id'], name='FKp6vjabv2e2435at1hnuxg64yv'),
        Index('FK86p3roa187k12avqfl28klp1q', 'genre_id'),
        Index('FKp6vjabv2e2435at1hnuxg64yv', 'movie_id')
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    genre_id: Mapped[Optional[int]] = mapped_column(Integer)
    movie_id: Mapped[Optional[int]] = mapped_column(Integer)

    genre: Mapped[Optional['Genre']] = relationship('Genre', back_populates='movie_genre')
    movie: Mapped[Optional['Movie']] = relationship('Movie', back_populates='movie_genre')


class MoviePeople(Base):
    __tablename__ = 'movie_people'
    __table_args__ = (
        ForeignKeyConstraint(['movie_id'], ['movie.id'], name='FKa65rtm8x60j3hw9jglrhl4ii6'),
        ForeignKeyConstraint(['people_id'], ['people.id'], name='FKiiehyvb6wrwfl1mefsl3ywwc8'),
        Index('FKa65rtm8x60j3hw9jglrhl4ii6', 'movie_id'),
        Index('FKiiehyvb6wrwfl1mefsl3ywwc8', 'people_id')
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    movie_id: Mapped[Optional[int]] = mapped_column(Integer)
    people_id: Mapped[Optional[int]] = mapped_column(Integer)

    movie: Mapped[Optional['Movie']] = relationship('Movie', back_populates='movie_people')
    people: Mapped[Optional['People']] = relationship('People', back_populates='movie_people')


class Review(Base):
    __tablename__ = 'review'
    __table_args__ = (
        ForeignKeyConstraint(['movie_id'], ['movie.id'], name='FK8378dhlpvq0e9tnkn9mx0r64i'),
        Index('FK8378dhlpvq0e9tnkn9mx0r64i', 'movie_id')
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    movie_id: Mapped[Optional[int]] = mapped_column(Integer)
    create_at: Mapped[Optional[datetime.datetime]] = mapped_column(DATETIME(fsp=6))
    author: Mapped[Optional[str]] = mapped_column(String(255))
    comment: Mapped[Optional[str]] = mapped_column(String(255))

    movie: Mapped[Optional['Movie']] = relationship('Movie', back_populates='review')


class UserMovie(Base):
    __tablename__ = 'user_movie'
    __table_args__ = (
        ForeignKeyConstraint(['movie_id'], ['movie.id'], name='FKic2w099acvoapsiypkvkwio7n'),
        ForeignKeyConstraint(['user_id'], ['user.id'], name='FKkslwq0qn0h1waemnmedse5ncb'),
        Index('FKic2w099acvoapsiypkvkwio7n', 'movie_id'),
        Index('FKkslwq0qn0h1waemnmedse5ncb', 'user_id')
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    movie_id: Mapped[Optional[int]] = mapped_column(Integer)
    user_id: Mapped[Optional[int]] = mapped_column(Integer)

    movie: Mapped[Optional['Movie']] = relationship('Movie', back_populates='user_movie')
    user: Mapped[Optional['User']] = relationship('User', back_populates='user_movie')
